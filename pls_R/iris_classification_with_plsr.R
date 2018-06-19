source("http://pastebin.com/raw.php?i=UyDBTA57")

data(iris)
tmp_vars <-iris[,-5]
tmp_class <-iris[,5] # species
tmp_y <-matrix(as.numeric(tmp_class),ncol=1) # make numeric matrix of the Iris species

## run through plsr
m1 <- pls::plsr(tmp_y ~ as.matrix(tmp_vars))

predict(m1)

factor(tmp_y, labels = levels(tmp.group), levels = 1:3)
