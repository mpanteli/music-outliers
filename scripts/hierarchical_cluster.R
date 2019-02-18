library(ape)

# load data
df = read.csv("../data/cluster_freq.csv")
data = df[,2:dim(df)[2]]

# shorten country labels for plotting
levels(df$labels)[which(levels(df$labels)=="Democratic Republic of the Congo")]="DR Congo"
df$labels[which(df$labels=="Democratic Republic of the Congo")] = "DR Congo"
rownames(data) <- df$labels

# hierarchical clustering based on Mahalanobis
distMahal = as.dist(apply(data, 1, function(i) mahalanobis(data, i, cov = cov(data),tol=1e-18)))
hc=hclust(distMahal, method="average")
mypal = c("#000000", "#9B0000", "#9B0000", "#9B0000", "#9B0000")
clus5 = cutree(hc, 4)

# plot and output figure
pdf('../data/hierarchical_cluster.pdf', pointsize=12)
par(mar=c(1,1,1,1))
plot(as.phylo(hc),type="fan",tip.color=mypal[clus5], cex=.5, label.offset=.5)
dev.off()
