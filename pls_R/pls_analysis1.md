NLCD\_PLS1
================
Cody Flagg
June 19, 2018

### Partial Least Squares (PLS) Analysis of NLCD data from SCBI

-   This doesn't split data into training and test sets
-   Only focuses on predicting 3 NLCD classes: "DF", "EF", "SS"

``` r
## plot raw data
wavelengths<-seq(1, 426,by=1) # wavelengths captured by machine
reflectance <- subdata[,c(11:ncol(subdata))] # reflectance values 

# need to setup the x-axis, the wavelengths, and a transpose of the reflectance matrix with t()
reflectance.plot = matplot(x = wavelengths, y = t(reflectance),lty=1,xlab="wavelengths(nm)",ylab="Reflectance",type="l", main = "Reflectance by Sample")
```

![](pls_analysis1_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
## convert reflectance to a matrix, as this is what plsr() requires
refl = as.matrix(reflectance)
subdata$NLCD <- as.factor(subdata$NLCD)

## convert NLCD character to factor first, then convert the factor to a numeric value so plsr() can analyze data
NLCD_num <- as.matrix(as.numeric(as.factor(subdata$NLCD)),ncol=1) # make numeric matrix

## run the model
m1 <- plsr(NLCD_num ~ log(refl+1), ncomp = 100, validation = "CV")

## these are the numeric classes converted back to NLCD labels
NLCD_back <-factor(NLCD_num, labels = levels(subdata$NLCD), levels = 1:3)

## add predictions from the plsr() model
subdata$pred_num_class <- predict(m1, ncomp = 25) # the output of this is an array, each column in the array is the predicted output for each principal component
#subdata$pred_fac_class <- round(subdata$pred_num_class)

## 1. take the mean prediction from the components (specified by 'ncomp') by applying mean() across a row
## 2. then round that mean, it represents a "vote" for the class value from each component
confusion <- data.frame(actual = NLCD_num, predicted = round(apply(X = subdata$pred_num_class, 1, FUN = mean)))

# cross reference the actual class versus the predicted class
crossRef <- table(confusion)
```

### Total Components Selected

-   The root mean squared error "RMSE" plot shows that ~25 principal components returns the lowest amount of error, thus all following graphs and confusion tables use 25 PCs

``` r
barplot(explvar(m1), axis.lty = 1, las = 2, main = "% Variance Explained by Component")
```

![](pls_analysis1_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r
plot(RMSEP(m1), legendpos = "topright", main = "Validation Error")
```

![](pls_analysis1_files/figure-markdown_github/unnamed-chunk-3-2.png)

### Loadings for 25 Components - untransformed Y and X

``` r
loadingplot(m1, comps = 1:25, legendpos = -1, labels = c(1:426), xlab = "Band")
```

![](pls_analysis1_files/figure-markdown_github/unnamed-chunk-4-1.png)

### Confusion Matrix -- Full Data Set (not split between training and test sets)

-   A fair amount of misclassification with DF
-   Another predictor variable that helps the model differentiate between DF and EF would probably reduce error e.g. do DF and EF occur at different elevations?

``` r
# calculate accuracy for each class
## subset the crossRef table because it has classes predicted as 0 and 4 <for 3 class>
if (ncol(crossRef != length(unique(NLCD_num)))){
    caret::confusionMatrix(crossRef[,c(2:4)])
} else {
    caret::confusionMatrix(crossRef)
}
```

    ## Confusion Matrix and Statistics
    ## 
    ##       predicted
    ## actual   1   2   3
    ##      1 208  82   1
    ##      2   1  25  11
    ##      3   0 100  58
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.599         
    ##                  95% CI : (0.554, 0.643)
    ##     No Information Rate : 0.43          
    ##     P-Value [Acc > NIR] : 5.81e-14      
    ##                                         
    ##                   Kappa : 0.395         
    ##  Mcnemar's Test P-Value : < 2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3
    ## Sensitivity             0.995   0.1208    0.829
    ## Specificity             0.700   0.9570    0.760
    ## Pos Pred Value          0.715   0.6757    0.367
    ## Neg Pred Value          0.995   0.5947    0.963
    ## Prevalence              0.430   0.4259    0.144
    ## Detection Rate          0.428   0.0514    0.119
    ## Detection Prevalence    0.599   0.0761    0.325
    ## Balanced Accuracy       0.848   0.5389    0.794
