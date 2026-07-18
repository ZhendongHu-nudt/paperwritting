# =============================================================
# RAR -- Build system
# Requires: pdflatex, bibtex (TeX Live / MacTeX / MikTeX)
#
# Targets:
#   make          -> paper.pdf  (multi-pass, default)
#   make quick    -> single pdflatex pass (no bib refresh)
#   make clean    -> remove build artifacts
#   make distclean-> clean + remove paper.pdf
#
# Note: TikZ figures are \input{}-style (figures/fig_*.tex),
# so no separate figure compilation is required.
# =============================================================

LATEX  := pdflatex
BIBTEX := bibtex
PAPER  := paper

.PHONY: all quick clean distclean

all: $(PAPER).pdf

# Standard four-pass build: latex -> bibtex -> latex -> latex.
$(PAPER).pdf: $(PAPER).tex \
             abstract.tex intro_final.tex background.tex method.tex \
             eval_scaffold.tex related_work.tex conclusion.tex \
             references.bib \
             figures/fig_rar_architecture.tex \
             figures/fig_recon_routing.tex \
             figures/fig_cil_setup.tex
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER)

quick: $(PAPER).tex
	$(LATEX) -interaction=nonstopmode $(PAPER)

clean:
	-@rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
            $(PAPER).out $(PAPER).toc $(PAPER).fls $(PAPER).fdb_latexmk \
            $(PAPER).synctex.gz $(PAPER).run.xml

distclean: clean
	-@rm -f $(PAPER).pdf