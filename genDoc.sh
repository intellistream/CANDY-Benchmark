doxygen Doxyfile
cp figures/documentSource/*.png doc/latex
cp figures/documentSource/*.pdf doc/latex
cd doc/latex && make && cp refman.pdf ../..
