"""
Reconstruction-gated dual-head incremental ET-BERT trainer.

Key idea:
- Fit an old-feature reconstructor on replay memory only.
- Use reconstruction error as an old/new score instead of an explicit is_old flag.
- Add the old/new score as a soft routing prior over the two classifier heads.
"""
import argparse
import json
import os
from datetime import datetime

# GPU selection must happen before torch import.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", "-g", type=str, default=None)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import BertConfig, Trainer

try:
    from improved_LwF_bert_trainer import (
        ETBERT_PRETRAINED_PATH,
        ETBERT_TOKENIZER_PATH,
        ResidualAdapter,
        _ETBertBackbone,
        _ETBertTeacherModel,
        _compute_detailed_metrics,
        _count_trainable_parameters,
        _func_bibytes,
        _get_tokenizer,
        _log_model_runtime_device,
        _make_training_args,
        _resolve_old_model_dir,
    )
except ImportError:
    from better_flowmlab.modelkit.improved_LwF_bert_trainer import (
        ETBERT_PRETRAINED_PATH,
        ETBERT_TOKENIZER_PATH,
        ResidualAdapter,
        _ETBertBackbone,
        _ETBertTeacherModel,
        _compute_detailed_metrics,
        _count_trainable_parameters,
        _func_bibytes,
        _get_tokenizer,
        _log_model_runtime_device,
        _make_training_args,
        _resolve_old_model_dir,
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_replay_train_df(df_new, df_old_memory, replay_ratio, seed=42):
    new_block = df_new.copy()
    if df_old_memory is None or len(df_old_memory) == 0:
        return new_block.sample(frac=1, random_state=seed).reset_index(drop=True), 0

    target_old = max(len(df_old_memory), int(len(df_new) * replay_ratio))
    old_replay = df_old_memory.sample(
        n=target_old,
        replace=(target_old > len(df_old_memory)),
        random_state=seed,
    ).copy()

    df_train = pd.concat([new_block, old_replay], axis=0).reset_index(drop=True)
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_train, len(old_replay)


class FeatureReconstructor(nn.Module):
    def __init__(self, hidden_size, bottleneck_dim, dropout=0.1):
        super().__init__()
        bottleneck_dim = min(hidden_size, max(8, bottleneck_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_size),
        )

    def forward(self, hidden):
        return self.net(hidden)


class ReconstructionGatedDualHeadETBertModel(nn.Module):
    def __init__(self, bert_config, old_classes, new_classes, adapter_dim=128, recon_dim=256):
        super().__init__()
        self.backbone = _ETBertBackbone(bert_config)
        self.old_classifier = nn.Linear(bert_config.hidden_size, old_classes) if old_classes > 0 else None
        self.adapter = ResidualAdapter(bert_config.hidden_size, adapter_dim)
        self.new_norm = nn.LayerNorm(bert_config.hidden_size)
        self.new_mlp = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.new_classifier = nn.Linear(bert_config.hidden_size, new_classes) if new_classes > 0 else None
        self.reconstructor = FeatureReconstructor(bert_config.hidden_size, recon_dim, dropout=0.1)
        self.old_logit_bias = nn.Parameter(torch.tensor(0.0))
        self.new_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.new_logit_bias = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("recon_threshold", torch.tensor(0.0))
        self.register_buffer("recon_temperature", torch.tensor(1.0))

    def forward(self, inputs):
        batch_size = inputs["raw"].size(0)
        cls_hidden = self.backbone(inputs)
        adapted_hidden = self.adapter(cls_hidden)
        new_hidden = self.new_mlp(self.new_norm(adapted_hidden))

        reconstructed_hidden = self.reconstructor(cls_hidden)
        recon_error = (reconstructed_hidden - cls_hidden).pow(2).mean(dim=1)
        scaled_temperature = self.recon_temperature.clamp_min(1e-6)

        if self.old_classifier is not None:
            old_score = torch.sigmoid((self.recon_threshold - recon_error) / scaled_temperature)
            raw_old_logits = self.old_classifier(cls_hidden) - self.old_logit_bias
            gated_old_logits = raw_old_logits + torch.log(old_score.clamp_min(1e-6)).unsqueeze(1)
        else:
            old_score = torch.zeros_like(recon_error)
            raw_old_logits = cls_hidden.new_zeros(batch_size, 0)
            gated_old_logits = raw_old_logits

        if self.new_classifier is not None:
            new_score = 1.0 - old_score
            raw_new_logits = self.new_classifier(new_hidden) * self.new_logit_scale + self.new_logit_bias
            gated_new_logits = raw_new_logits + torch.log(new_score.clamp_min(1e-6)).unsqueeze(1)
        else:
            if self.old_classifier is not None:
                old_score = torch.ones_like(old_score)
            new_score = torch.zeros_like(old_score)
            raw_new_logits = cls_hidden.new_zeros(batch_size, 0)
            gated_new_logits = raw_new_logits

        combined_logits = torch.cat([gated_old_logits, gated_new_logits], dim=1)
        return {
            "y_logit": combined_logits,
            "old_logits": raw_old_logits,
            "new_logits": raw_new_logits,
            "cls_hidden": cls_hidden,
            "adapted_hidden": adapted_hidden,
            "new_hidden": new_hidden,
            "reconstructed_hidden": reconstructed_hidden,
            "recon_error": recon_error,
            "old_score": old_score,
            "new_score": new_score,
        }


class ReconstructionGatedTrainer(Trainer):
    def __init__(
        self,
        teacher_model=None,
        old_out_dim=None,
        alpha_old_distill=0.1,
        alpha_old_feature=0.0,
        alpha_new_margin=0.3,
        alpha_adapter_reg=0.02,
        alpha_new_local=0.3,
        distill_temperature=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.old_out_dim = old_out_dim
        self.alpha_old_distill = alpha_old_distill
        self.alpha_old_feature = alpha_old_feature
        self.alpha_new_margin = alpha_new_margin
        self.alpha_adapter_reg = alpha_adapter_reg
        self.alpha_new_local = alpha_new_local
        self.distill_temperature = distill_temperature

        if self.teacher is not None:
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False

        print("\n[Recon-gated dual-head] Initialized with:")
        print(f"  alpha_old_distill={alpha_old_distill}")
        print(f"  alpha_old_feature={alpha_old_feature}")
        print(f"  alpha_new_margin={alpha_new_margin}")
        print(f"  alpha_adapter_reg={alpha_adapter_reg}")
        print(f"  alpha_new_local={alpha_new_local}")
        print(f"  distill_temperature={distill_temperature}")

    def _per_sample_kl(self, student_logits, teacher_logits):
        temperature = self.distill_temperature
        student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
        teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * (temperature ** 2)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(inputs)
        logits = outputs["y_logit"]
        old_logits = outputs["old_logits"]
        new_logits = outputs["new_logits"]
        cls_hidden = outputs["cls_hidden"]
        adapted_hidden = outputs["adapted_hidden"]
        new_hidden = outputs["new_hidden"]
        labels = inputs["y_label"]
        old_score = outputs["old_score"]
        new_score = outputs["new_score"]

        loss = F.cross_entropy(logits, labels)

        if self.teacher is not None and old_logits.size(1) > 0:
            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)
                teacher_logits = teacher_outputs["y_logit"]
                teacher_hidden = teacher_outputs["cls_hidden"]

            old_weight = old_score
            old_weight_norm = old_weight.sum().clamp_min(1e-6)
            distill_loss = self._per_sample_kl(old_logits, teacher_logits)
            loss = loss + self.alpha_old_distill * (old_weight * distill_loss).sum() / old_weight_norm

            if self.alpha_old_feature > 0:
                feature_old = (adapted_hidden - teacher_hidden).pow(2).mean(dim=1)
                loss = loss + self.alpha_old_feature * (old_weight * feature_old).sum() / old_weight_norm

        if self.old_out_dim is not None and new_logits.size(1) > 0:
            new_label_mask = labels >= self.old_out_dim
            if new_label_mask.any():
                new_labels = labels[new_label_mask] - self.old_out_dim
                new_logits_only = new_logits[new_label_mask]
                local_new_ce = F.cross_entropy(new_logits_only, new_labels)
                loss = loss + self.alpha_new_local * local_new_ce

                if old_logits.size(1) > 0:
                    old_logits_on_new = old_logits[new_label_mask]
                    correct_new_logit = new_logits_only.gather(1, new_labels.unsqueeze(1)).squeeze(1)
                    max_old_on_new = old_logits_on_new.max(dim=1)[0]
                    margin_loss = torch.relu(max_old_on_new - correct_new_logit + self.alpha_new_margin)
                    new_weight = new_score[new_label_mask]
                    new_weight_norm = new_weight.sum().clamp_min(1e-6)
                    loss = loss + 0.1 * (new_weight * margin_loss).sum() / new_weight_norm

                adapter_shift = (new_hidden[new_label_mask] - cls_hidden[new_label_mask]).pow(2).mean()
                loss = loss + self.alpha_adapter_reg * adapter_shift

        return (loss, outputs) if return_outputs else loss


class ETBertImprovedLwF1Model:
    model_type_name = "ET-BERT"

    def __init__(self, max_length=512):
        self.max_length = max_length
        self.label2idx = None
        self.old_out_dim = None

    def _build_config(self):
        with open(os.path.join(ETBERT_PRETRAINED_PATH, "hyperparameters.json")) as handle:
            hyperparameters = json.load(handle)
        tokenizer = _get_tokenizer(ETBERT_TOKENIZER_PATH)
        return BertConfig(
            vocab_size=len(tokenizer.get_vocab()),
            pad_token_id=tokenizer.pad_token_id,
            hidden_size=hyperparameters["d_model"],
            num_hidden_layers=hyperparameters["n_layer"],
            num_attention_heads=hyperparameters["n_head"],
            intermediate_size=hyperparameters["dim_ff"],
            max_position_embeddings=hyperparameters["max_length"],
        )

    def create_model(self, old_classes, new_classes, adapter_dim, recon_dim):
        return ReconstructionGatedDualHeadETBertModel(
            self._build_config(),
            old_classes,
            new_classes,
            adapter_dim=adapter_dim,
            recon_dim=recon_dim,
        )

    def create_teacher_model(self, old_classes):
        return _ETBertTeacherModel(self._build_config(), old_classes)

    def preprocess_data(self, df):
        tokenizer = _get_tokenizer(ETBERT_TOKENIZER_PATH)
        fwd = df["fwd_raw"].fillna("(empty)").apply(_func_bibytes)
        bwd = df["bwd_raw"].fillna("(empty)").apply(_func_bibytes)
        encodings = tokenizer(
            list(zip(fwd.tolist(), bwd.tolist())),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_token_type_ids=True,
        )

        labels = df["label"].astype(str)
        y_label = labels.map(self.label2idx).fillna(-1).astype(int).tolist()

        valid_indices = [index for index, label in enumerate(y_label) if label != -1]
        if len(valid_indices) < len(y_label):
            encodings["input_ids"] = [encodings["input_ids"][i] for i in valid_indices]
            encodings["attention_mask"] = [encodings["attention_mask"][i] for i in valid_indices]
            encodings["token_type_ids"] = [encodings["token_type_ids"][i] for i in valid_indices]
            y_label = [y_label[i] for i in valid_indices]

        dataset = Dataset.from_dict(
            {
                "raw": encodings["input_ids"],
                "raw_attention_mask": encodings["attention_mask"],
                "raw_token_type_ids": encodings["token_type_ids"],
                "y_label": y_label,
            }
        )
        dataset.set_format(
            type="torch",
            columns=["raw", "raw_attention_mask", "raw_token_type_ids", "y_label"],
        )
        return dataset

    def _iterate_batches(self, dataset, batch_size, shuffle=False):
        if dataset is None or dataset.num_rows == 0:
            return

        indices = np.arange(dataset.num_rows)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size].tolist()
            batch = dataset[batch_indices]
            yield {key: value.to(DEVICE) for key, value in batch.items()}

    def evaluate_model(self, model, testset, args, only_old=False):
        model = model.to(DEVICE)
        model.eval()
        batch_size = getattr(args, "batch_size", 32)
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch in tqdm(self._iterate_batches(testset, batch_size, shuffle=False), desc="Evaluating"):
                logits = model(batch)["y_logit"]
                if only_old and self.old_out_dim is not None:
                    logits = logits[:, :self.old_out_dim]
                y_pred_tmp = logits.argmax(1).cpu().numpy()
                y_true_tmp = batch["y_label"].cpu().numpy()
                y_pred = np.concatenate([y_pred, y_pred_tmp]) if len(y_pred) else y_pred_tmp
                y_true = np.concatenate([y_true, y_true_tmp]) if len(y_true) else y_true_tmp
        return y_pred, y_true

    def _collect_reconstruction_errors(self, model, dataset, args):
        if dataset is None or dataset.num_rows == 0:
            return np.array([], dtype=np.float32)

        model = model.to(DEVICE)
        model.eval()
        batch_size = getattr(args, "batch_size", 32)
        errors = []
        with torch.no_grad():
            for batch in self._iterate_batches(dataset, batch_size, shuffle=False):
                cls_hidden = model.backbone(batch)
                reconstructed_hidden = model.reconstructor(cls_hidden)
                batch_error = (reconstructed_hidden - cls_hidden).pow(2).mean(dim=1)
                errors.append(batch_error.cpu())

        if not errors:
            return np.array([], dtype=np.float32)
        return torch.cat(errors).numpy()

    def _fit_reconstructor(self, model, old_memoryset, args, new_refset=None):
        if self.old_out_dim is None or self.old_out_dim <= 0 or old_memoryset is None or old_memoryset.num_rows == 0:
            print("\nSkipping reconstruction gate fitting because old replay memory is unavailable.")
            with torch.no_grad():
                model.recon_threshold.fill_(0.0)
                model.recon_temperature.fill_(1.0)
            for parameter in model.reconstructor.parameters():
                parameter.requires_grad = False
            return

        print("\nFitting old-feature reconstructor on replay memory...")
        model = model.to(DEVICE)
        model.reconstructor.train()
        for parameter in model.reconstructor.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.reconstructor.parameters(),
            lr=getattr(args, "recon_lr", 1e-3),
            weight_decay=getattr(args, "recon_weight_decay", 1e-4),
        )

        batch_size = getattr(args, "batch_size", 32)
        n_epochs = getattr(args, "recon_epochs", 5)
        for epoch in range(n_epochs):
            epoch_losses = []
            for batch in self._iterate_batches(old_memoryset, batch_size, shuffle=True):
                optimizer.zero_grad()
                with torch.no_grad():
                    cls_hidden = model.backbone(batch)
                reconstructed_hidden = model.reconstructor(cls_hidden.detach())
                loss = F.mse_loss(reconstructed_hidden, cls_hidden.detach())
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            if epoch_losses:
                print(f"  Recon epoch {epoch + 1}/{n_epochs}: loss={np.mean(epoch_losses):.6f}")

        model.reconstructor.eval()
        old_errors = self._collect_reconstruction_errors(model, old_memoryset, args)
        new_errors = self._collect_reconstruction_errors(model, new_refset, args)

        old_quantile = getattr(args, "recon_old_quantile", 0.95)
        if new_errors.size > 0:
            old_edge = float(np.quantile(old_errors, old_quantile))
            new_edge = float(np.quantile(new_errors, max(0.0, 1.0 - old_quantile)))
            threshold = 0.5 * (old_edge + new_edge)
            spread = float(np.std(np.concatenate([old_errors, new_errors])))
        else:
            threshold = float(np.quantile(old_errors, old_quantile))
            spread = float(np.std(old_errors))

        temperature = max(spread * getattr(args, "recon_temperature_scale", 1.0), 1e-4)
        with torch.no_grad():
            model.recon_threshold.fill_(threshold)
            model.recon_temperature.fill_(temperature)

        print(f"  Old recon error mean/std: {old_errors.mean():.6f} / {old_errors.std():.6f}")
        if new_errors.size > 0:
            print(f"  New recon error mean/std: {new_errors.mean():.6f} / {new_errors.std():.6f}")
        print(f"  Recon threshold: {threshold:.6f}")
        print(f"  Recon temperature: {temperature:.6f}")

        for parameter in model.reconstructor.parameters():
            parameter.requires_grad = False

    def _calibrate(self, model, calibset, args):
        print("\nCalibrating old/new logits on mixed validation set...")
        model = model.to(DEVICE)
        model.eval()

        params = [model.old_logit_bias, model.new_logit_scale, model.new_logit_bias]
        optimizer = torch.optim.Adam(params, lr=1e-2)
        batch_size = getattr(args, "batch_size", 32)

        for _ in range(getattr(args, "calibration_steps", 100)):
            optimizer.zero_grad()
            total_loss = 0.0
            for batch in self._iterate_batches(calibset, batch_size, shuffle=False):
                outputs = model(batch)
                total_loss = total_loss + F.cross_entropy(outputs["y_logit"], batch["y_label"])
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.new_logit_scale.data.clamp_(min=0.5, max=3.0)

    def train(self, df_new, df_test, model_dir, timestamp, args, df_eval=None, df_old=None):
        print("\n" + "=" * 80)
        print("Training ET-BERT model with Improved LwF1 (reconstruction-gated)...")
        print("=" * 80)

        df_new = df_new.copy()
        df_test = df_test.copy()
        df_new["label"] = df_new["label"].astype(str)
        df_test["label"] = df_test["label"].astype(str)
        if df_eval is not None:
            df_eval = df_eval.copy()
            df_eval["label"] = df_eval["label"].astype(str)
        if df_old is not None:
            df_old = df_old.copy()
            df_old["label"] = df_old["label"].astype(str)

        os.makedirs(model_dir, exist_ok=True)

        old_model_dir = _resolve_old_model_dir(getattr(args, "old_model_dir", ""), self.model_type_name)
        if old_model_dir is not None:
            label_path = os.path.join(old_model_dir, "label2idx.json")
            print(f"Loading old label mapping: {label_path}")
            with open(label_path, "r") as handle:
                loaded_mapping = json.load(handle)
            self.label2idx = {str(key): value for key, value in loaded_mapping.items()}
            print(f"Old classes: {list(self.label2idx.keys())}")
        else:
            print("No old mapping found, creating new mapping")
            self.label2idx = {}

        old_class_num = len(self.label2idx)
        for label in pd.concat([df_new["label"]]).unique():
            if label not in self.label2idx:
                self.label2idx[label] = len(self.label2idx)
        num_classes = len(self.label2idx)
        new_class_num = num_classes - old_class_num

        print("\nClass statistics:")
        print(f"  Old classes: {old_class_num}")
        print(f"  New classes: {new_class_num}")
        print(f"  Total classes: {num_classes}")

        df_old_memory = getattr(args, "_old_memory_df", None)
        df_balanced_train, replay_count = _build_replay_train_df(
            df_new,
            df_old_memory,
            getattr(args, "replay_ratio", 1.0),
            seed=42,
        )
        print(f"  Replay old samples used in training dataframe: {replay_count}")
        print(f"  Balanced train size: {len(df_balanced_train)}")

        with open(os.path.join(model_dir, "label2idx.json"), "w") as handle:
            json.dump(self.label2idx, handle, indent=2)

        trainset = self.preprocess_data(df_balanced_train)
        eval_source_df = df_eval if df_eval is not None else df_balanced_train
        evalset = self.preprocess_data(eval_source_df)
        testset_new = self.preprocess_data(df_test)
        old_memoryset = None
        if df_old_memory is not None and len(df_old_memory) > 0:
            old_memoryset = self.preprocess_data(df_old_memory.reset_index(drop=True))
        new_refset = None
        if df_eval is not None and len(df_eval) > 0:
            new_refset = self.preprocess_data(df_eval)

        oldset = None
        mixed_calib_df = None
        if df_old is not None:
            print("\nPreparing evaluation datasets...")
            oldset = self.preprocess_data(df_old)
            if df_old_memory is not None and df_eval is not None and len(df_eval) > 0:
                old_calib_n = min(len(df_old_memory), len(df_eval))
                old_calib_df = df_old_memory.sample(n=old_calib_n, random_state=42)
                mixed_calib_df = pd.concat([df_eval, old_calib_df], axis=0).reset_index(drop=True)

        checkpoint_path = os.path.join(old_model_dir, "pytorch_model.bin") if old_model_dir else None
        teacher_model = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\nLoading pretrained model: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.old_out_dim = state_dict["classifier.weight"].shape[0]
            print(f"  Old classes from model: {self.old_out_dim}")

            teacher_model = self.create_teacher_model(self.old_out_dim)
            missing, unexpected = teacher_model.load_state_dict(state_dict, strict=False)
            print(f"  Teacher missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            teacher_model = teacher_model.to(DEVICE)
            teacher_model.eval()

            model = self.create_model(
                self.old_out_dim,
                new_class_num,
                getattr(args, "adapter_dim", 256),
                getattr(args, "recon_dim", 256),
            )
            backbone_state = {f"backbone.{k}": v for k, v in state_dict.items() if k.startswith("bert.")}
            model_state = model.state_dict()
            model_state.update(backbone_state)
            if model.old_classifier is not None:
                model_state["old_classifier.weight"] = state_dict["classifier.weight"]
                model_state["old_classifier.bias"] = state_dict["classifier.bias"]
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            print(f"  Model missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        else:
            print("\n⚠️ Pretrained model not found, training from ET-BERT pretrained weights...")
            model = self.create_model(
                old_class_num,
                new_class_num,
                getattr(args, "adapter_dim", 256),
                getattr(args, "recon_dim", 256),
            )
            model.backbone.load_pretrained()
            teacher_model = None
            self.old_out_dim = old_class_num if old_class_num > 0 else None

        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        if model.old_classifier is not None:
            for parameter in model.old_classifier.parameters():
                parameter.requires_grad = False

        self._fit_reconstructor(model, old_memoryset, args, new_refset=new_refset)

        model = model.to(DEVICE)
        _log_model_runtime_device(self.model_type_name, model)

        eval1_old_before_train = {}
        if oldset is not None and self.old_out_dim is not None and self.old_out_dim > 0:
            print("\n" + "=" * 60)
            print("[Evaluation 1] Before training - Old test set performance")
            print("=" * 60)
            y_pred_old_before, y_true_old_before = self.evaluate_model(model, oldset, args, only_old=True)
            eval1_old_before_train = _compute_detailed_metrics(y_true_old_before, y_pred_old_before)
            print(f"  Old Accuracy: {eval1_old_before_train['accuracy']:.4f}")
            print(f"  Old Macro-F1: {eval1_old_before_train['macro_f1']:.4f}")

        print("\n" + "=" * 60)
        print("Training reconstruction-gated adapter + new head with frozen old path")
        print("=" * 60)
        trainable, total = _count_trainable_parameters(model)
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

        trainer = ReconstructionGatedTrainer(
            model=model,
            teacher_model=teacher_model,
            old_out_dim=self.old_out_dim,
            alpha_old_distill=getattr(args, "alpha_old_distill", 0.1),
            alpha_old_feature=getattr(args, "alpha_old_feature", 0.0),
            alpha_new_margin=getattr(args, "alpha_new_margin", 0.3),
            alpha_adapter_reg=getattr(args, "alpha_adapter_reg", 0.02),
            alpha_new_local=getattr(args, "alpha_new_local", 0.3),
            distill_temperature=getattr(args, "distill_temperature", 2.0),
            args=_make_training_args(
                model_dir,
                getattr(args, "learning_rate", 2e-4),
                getattr(args, "batch_size", 32),
                getattr(args, "n_epochs", 20),
                getattr(args, "warmup_ratio", 0.05),
                getattr(args, "weight_decay", 1e-4),
                ["y_label"],
            ),
            train_dataset=trainset,
            eval_dataset=evalset,
        )

        if not getattr(args, "no_train", False):
            trainer.train()

        if mixed_calib_df is not None and len(mixed_calib_df) > 0:
            calibset = self.preprocess_data(mixed_calib_df)
            self._calibrate(model, calibset, args)

        if not getattr(args, "no_train", False):
            #torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
            trainer.save_state()

        eval2_old_after_train = {}
        if oldset is not None and self.old_out_dim is not None and self.old_out_dim > 0:
            print("\n" + "=" * 60)
            print("[Evaluation 2] After training - Old test set performance")
            print("=" * 60)
            y_pred_old_after, y_true_old_after = self.evaluate_model(model, oldset, args, only_old=False)
            eval2_old_after_train = _compute_detailed_metrics(y_true_old_after, y_pred_old_after)
            print(f"  Old Accuracy: {eval2_old_after_train['accuracy']:.4f}")
            print(f"  Old Macro-F1: {eval2_old_after_train['macro_f1']:.4f}")

        print("\n" + "=" * 60)
        print("[Evaluation 3] After training - New test set performance")
        print("=" * 60)
        y_pred_new, y_true_new = self.evaluate_model(model, testset_new, args, only_old=False)
        eval3_new_after_train = _compute_detailed_metrics(y_true_new, y_pred_new)
        print(f"  New Accuracy: {eval3_new_after_train['accuracy']:.4f}")
        print(f"  New Macro-F1: {eval3_new_after_train['macro_f1']:.4f}")

        eval4_mixed_after_train = {}
        if df_old is not None:
            print("\n" + "=" * 60)
            print("[Evaluation 4] After training - Mixed test set performance")
            print("=" * 60)
            df_mixed = pd.concat([df_test, df_old], axis=0).reset_index(drop=True)
            mixedset = self.preprocess_data(df_mixed)
            y_pred_mixed, y_true_mixed = self.evaluate_model(model, mixedset, args, only_old=False)
            eval4_mixed_after_train = _compute_detailed_metrics(y_true_mixed, y_pred_mixed)
            print(f"  Mixed Overall Accuracy: {eval4_mixed_after_train['accuracy']:.4f}")
            print(f"  Mixed Macro-F1: {eval4_mixed_after_train['macro_f1']:.4f}")
            if self.old_out_dim is not None:
                old_mask = y_true_mixed < self.old_out_dim
                new_mask = y_true_mixed >= self.old_out_dim
                if old_mask.any():
                    print(f"  Mixed Old Part ({old_mask.sum()} samples): {accuracy_score(y_true_mixed[old_mask], y_pred_mixed[old_mask]):.4f}")
                if new_mask.any():
                    print(f"  Mixed New Part ({new_mask.sum()} samples): {accuracy_score(y_true_mixed[new_mask], y_pred_mixed[new_mask]):.4f}")

        forgetting = 0.0
        if eval1_old_before_train and eval2_old_after_train:
            forgetting = eval1_old_before_train["accuracy"] - eval2_old_after_train["accuracy"]

        result = {
            "type": self.model_type_name,
            "method": "Improved-LwF1-Recon",
            "num_old_classes": self.old_out_dim if self.old_out_dim else 0,
            "num_new_classes": new_class_num,
            "num_total_classes": num_classes,
            "reconstruction_gate": {
                "threshold": float(model.recon_threshold.item()),
                "temperature": float(model.recon_temperature.item()),
            },
            "eval1_old_before_train": eval1_old_before_train,
            "eval2_old_after_train": eval2_old_after_train,
            "eval3_new_after_train": eval3_new_after_train,
            "eval4_mixed_after_train": eval4_mixed_after_train,
            "incremental_metrics": {
                "forgetting": forgetting,
                "learning_accuracy": eval3_new_after_train.get("accuracy", 0),
                "average_accuracy": eval4_mixed_after_train.get("accuracy", 0),
                "backward_transfer": eval2_old_after_train.get("accuracy", 0) - eval1_old_before_train.get("accuracy", 0)
                if eval1_old_before_train and eval2_old_after_train else 0,
            },
        }

        print("\n" + "=" * 60)
        print("ET-BERT (Improved LwF1 Reconstruction-Gated) Training Complete!")
        print(f"  Forgetting: {result['incremental_metrics']['forgetting']:.4f}")
        print(f"  Learning Accuracy: {result['incremental_metrics']['learning_accuracy']:.4f}")
        print(f"  Average Accuracy: {result['incremental_metrics']['average_accuracy']:.4f}")
        print("=" * 60)
        return result


def print_results_summary(results):
    print("\n" + "=" * 100)
    print("Improved LwF1 Reconstruction-Gated Incremental Learning Results Summary")
    print("=" * 100)
    print(f"{'Model':<12} {'Method':<24} {'Old-Before':<12} {'Old-After':<12} {'New':<12} {'Mixed':<12} {'Forgetting':<12} {'LearnAcc':<12}")
    print("-" * 100)
    for result in results:
        eval1 = result.get("eval1_old_before_train", {})
        eval2 = result.get("eval2_old_after_train", {})
        eval3 = result.get("eval3_new_after_train", {})
        eval4 = result.get("eval4_mixed_after_train", {})
        metrics = result.get("incremental_metrics", {})
        print(
            f"{result.get('type', 'Unknown'):<12} {result.get('method', 'Improved-LwF1-Recon'):<24} "
            f"{eval1.get('accuracy', 0):<12.4f} {eval2.get('accuracy', 0):<12.4f} "
            f"{eval3.get('accuracy', 0):<12.4f} {eval4.get('accuracy', 0):<12.4f} "
            f"{metrics.get('forgetting', 0):<12.4f} {metrics.get('learning_accuracy', 0):<12.4f}"
        )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Reconstruction-gated Improved LwF1 for ET-BERT")
    parser.add_argument("--timestamp", "-T", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default="saves/hzd_Bert_job/new/Improved_LwF1")
    parser.add_argument("--dataset_path", "-P", type=str, default="/store1/huzhendong26/data/Real_LabeledDataset/new")
    parser.add_argument("--dataset_name", "-D", type=str, default="CIC-IoT2023")
    parser.add_argument("--sample_n", "-N", type=int, default=2000)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--n_epochs", "-ep", type=int, default=20)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
    parser.add_argument("--adapter_dim", type=int, default=256)
    parser.add_argument("--recon_dim", type=int, default=256)
    parser.add_argument("--recon_epochs", type=int, default=5)
    parser.add_argument("--recon_lr", type=float, default=1e-3)
    parser.add_argument("--recon_weight_decay", type=float, default=1e-4)
    parser.add_argument("--recon_old_quantile", type=float, default=0.95)
    parser.add_argument("--recon_temperature_scale", type=float, default=1.0)
    parser.add_argument("--old_model_dir", type=str, default="saves/hzd_Bert_job/new/baseline/CIC-IoT2023/ET-BERT")
    parser.add_argument("--memory_ratio", type=float, default=0.01)
    parser.add_argument("--replay_ratio", type=float, default=0.5)
    parser.add_argument("--no_train", action="store_true", help="Skip training and only evaluate")
    parser.add_argument("--gpu", "-g", type=str, default="0")
    parser.add_argument("--alpha_old_distill", type=float, default=0.1)
    parser.add_argument("--alpha_old_feature", type=float, default=0.0)
    parser.add_argument("--alpha_new_margin", type=float, default=0.3)
    parser.add_argument("--alpha_adapter_reg", type=float, default=0.02)
    parser.add_argument("--alpha_new_local", type=float, default=0.3)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--calibration_steps", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: CUDA_VISIBLE_DEVICES={args.gpu}  DEVICE={DEVICE}")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d%H%M")
    output_root = args.output if (os.path.isabs(args.output) or args.output.startswith(f"saves{os.sep}") or args.output == "saves") else os.path.join("saves", args.output)
    ratio_dir = f"old_{args.memory_ratio:g}"
    model_root = os.path.join(output_root, ratio_dir)
    run_root = os.path.join(model_root, timestamp)
    os.makedirs(run_root, exist_ok=True)

    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    df_new = pd.read_csv(
        f"{args.dataset_path}/new_train_{args.sample_n}/new {args.dataset_name}.csv",
        index_col=0,
        low_memory=False,
    )
    df_old = pd.read_csv(
        f"{args.dataset_path}/old_train_{args.sample_n}/{args.dataset_name}.csv",
        index_col=0,
        low_memory=False,
    )
    df_old_test = pd.read_csv(
        f"{args.dataset_path}/old_test/{args.dataset_name}.csv",
        index_col=0,
        low_memory=False,
    )

    memory_size = int(len(df_old) * args.memory_ratio)
    if memory_size == 0:
        memory_size = min(50, len(df_old))
        print(f"⚠️ memory_ratio=0, adjusting to {memory_size} old samples for replay")

    df_old_memory = df_old.sample(n=min(memory_size, len(df_old)), random_state=42)
    args._old_memory_df = df_old_memory.copy()

    print("\nDataset statistics:")
    print(f"  New data: {len(df_new)}")
    print(f"  Old memory (raw): {len(df_old_memory)}")
    print(f"  Old test: {len(df_old_test)}")

    df_test = pd.read_csv(
        f"{args.dataset_path}/new_test/new {args.dataset_name}.csv",
        index_col=0,
        low_memory=False,
    )
    df_eval = df_test.sample(frac=0.1, random_state=42)
    df_test = df_test.drop(df_eval.index).reset_index(drop=True)
    df_eval = df_eval.reset_index(drop=True)
    print(f"  New test: {len(df_test)}")
    print(f"  Evaluation: {len(df_eval)}")

    info = {
        "dataset": args.dataset_name,
        "sample_n": args.sample_n,
        "timestamp": timestamp,
        "new_data_size": len(df_new),
        "old_memory_size": len(df_old_memory),
        "old_test_size": len(df_old_test),
        "new_test_size": len(df_test),
        "eval_size": len(df_eval),
        "memory_ratio": args.memory_ratio,
        "replay_ratio": args.replay_ratio,
        "adapter_dim": args.adapter_dim,
        "recon_dim": args.recon_dim,
        "recon_epochs": args.recon_epochs,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "old_model_dir": args.old_model_dir,
    }
    with open(os.path.join(run_root, "info.json"), "w") as handle:
        json.dump(info, handle, indent=2)

    model = ETBertImprovedLwF1Model()
    model_dir = os.path.join(run_root, model.model_type_name)
    result = model.train(df_new, df_test, model_dir, timestamp, args, df_eval=df_eval, df_old=df_old_test)

    results = [result]
    with open(os.path.join(run_root, "training_results_lwf1.json"), "w") as handle:
        json.dump(results, handle, indent=2)

    print_results_summary(results)
    print(f"\n✅ Results saved to: {run_root}")


if __name__ == "__main__":
    main()
