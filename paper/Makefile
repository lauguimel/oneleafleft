# Makefile for paper/main.tex — ERL Letter
# Usage:
#   make          → compile PDF (pdflatex + bibtex + 2× pdflatex)
#   make clean    → remove build artifacts
#   make figures  → regenerate all figures from data
#
# NOTE: iopart.cls stub is included in paper/.
# For final submission, replace iopart.cls with the official version from:
#   https://ctan.org/pkg/iopart

MAIN    = main
LATEX   = pdflatex
BIBTEX  = bibtex
FIGS_SCRIPT = ../scripts/paper_figures.py

.PHONY: all clean figures watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib figures/fig1_study_area.pdf \
             figures/fig2_ablation.pdf figures/fig3_lift_shap.pdf
	$(LATEX) -interaction=nonstopmode $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN)

figures: $(FIGS_SCRIPT)
	conda run -n deforest python $(FIGS_SCRIPT)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).synctex.gz
