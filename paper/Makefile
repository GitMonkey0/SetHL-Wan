TECTONIC ?= tectonic

.PHONY: all package clean

all: main.pdf

main.pdf: main.tex refs.bib results.tex figures/method_overview.pdf figures/qualitative.pdf spconf.sty IEEEbib.bst
	$(TECTONIC) --keep-logs main.tex

package: main.pdf
	TECTONIC=$(TECTONIC) ./package_overleaf.sh

clean:
	rm -f main.aux main.bbl main.blg main.log main.out
