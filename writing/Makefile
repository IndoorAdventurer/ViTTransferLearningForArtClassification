
NAME=report
all: $(NAME).pdf

SECTIONS := $(shell echo ./tex/*)

$(NAME).pdf: $(NAME).tex literature.bib $(SECTIONS)
	pdflatex $(NAME)
	bibtex $(NAME).aux
	pdflatex $(NAME)
	clear
	pdflatex $(NAME)
	rm $(NAME).aux $(NAME).bbl  $(NAME).blg  $(NAME).log

clean:
	rm $(NAME).aux $(NAME).bbl  $(NAME).blg  $(NAME).log
	rm $(NAME).pdf
