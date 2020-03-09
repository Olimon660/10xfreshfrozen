library(Seurat)

#------pijuan-----------
D1A_FA3_THA = readRDS("./data/processed_counts/D1A_FA3_THA.rds")
D1X_FA2_FRE = readRDS("./data/processed_counts/D1X_FA2_FRE.rds")
D1X_FA2_THA = readRDS("./data/processed_counts/D1X_FA2_THA.rds")
D1X_FA3_FRE = readRDS("./data/processed_counts/D1X_FA3_FRE.rds")
D4X_FA2_FRE = readRDS("./data/processed_counts/D4X_FA2_FRE.rds")
D4X_FA2_THA = readRDS("./data/processed_counts/D4X_FA2_THA.rds")
D4X_FA3_FRE = readRDS("./data/processed_counts/D4X_FA3_FRE.rds")
D4X_FA3_THA = readRDS("./data/processed_counts/D4X_FA3_THA.rds")

D1A_FA3_THA.meta = as.data.frame(cbind(colnames(D1A_FA3_THA), "D1A_FA3_THA", "FROZEN"),
                                 stringsAsFactors = F)
colnames(D1A_FA3_THA.meta) = c("cell", "sample", "isFrozen")
rownames(D1A_FA3_THA.meta) = D1A_FA3_THA.meta$cell

D1X_FA2_FRE.meta = as.data.frame(cbind(colnames(D1X_FA2_FRE), "D1X_FA2_FRE", "FRESH"),
                                 stringsAsFactors = F)
colnames(D1X_FA2_FRE.meta) = c("cell", "sample", "isFrozen")
rownames(D1X_FA2_FRE.meta) = D1X_FA2_FRE.meta$cell

D1X_FA2_THA.meta = as.data.frame(cbind(colnames(D1X_FA2_THA), "D1X_FA2_THA", "FROZEN"),
                                 stringsAsFactors = F)
colnames(D1X_FA2_THA.meta) = c("cell", "sample", "isFrozen")
rownames(D1X_FA2_THA.meta) = D1X_FA2_THA.meta$cell

D1X_FA3_FRE.meta = as.data.frame(cbind(colnames(D1X_FA3_FRE), "D1X_FA3_FRE", "FRESH"),
                                 stringsAsFactors = F)
colnames(D1X_FA3_FRE.meta) = c("cell", "sample", "isFrozen")
rownames(D1X_FA3_FRE.meta) = D1X_FA3_FRE.meta$cell

D4X_FA2_FRE.meta = as.data.frame(cbind(colnames(D4X_FA2_FRE), "D4X_FA2_FRE", "FRESH"),
                                 stringsAsFactors = F)
colnames(D4X_FA2_FRE.meta) = c("cell", "sample", "isFrozen")
rownames(D4X_FA2_FRE.meta) = D4X_FA2_FRE.meta$cell

D4X_FA2_THA.meta = as.data.frame(cbind(colnames(D4X_FA2_THA), "D4X_FA2_THA", "FROZEN"),
                                 stringsAsFactors = F)
colnames(D4X_FA2_THA.meta) = c("cell", "sample", "isFrozen")
rownames(D4X_FA2_THA.meta) = D4X_FA2_THA.meta$cell

D4X_FA3_FRE.meta = as.data.frame(cbind(colnames(D4X_FA3_FRE), "D4X_FA3_FRE", "FRESH"),
                                 stringsAsFactors = F)
colnames(D4X_FA3_FRE.meta) = c("cell", "sample", "isFrozen")
rownames(D4X_FA3_FRE.meta) = D4X_FA3_FRE.meta$cell

D4X_FA3_THA.meta = as.data.frame(cbind(colnames(D4X_FA3_THA), "D4X_FA3_THA", "FROZEN"),
                                 stringsAsFactors = F)
colnames(D4X_FA3_THA.meta) = c("cell", "sample", "isFrozen")
rownames(D4X_FA3_THA.meta) = D4X_FA3_THA.meta$cell

sc.list = list("D1A_FA3_THA" = CreateSeuratObject(D1A_FA3_THA, meta.data = D1A_FA3_THA.meta),
               "D1X_FA2_FRE" = CreateSeuratObject(D1X_FA2_FRE, meta.data = D1X_FA2_FRE.meta),
               "D1X_FA2_THA" = CreateSeuratObject(D1X_FA2_THA, meta.data = D1X_FA2_THA.meta),
               "D1X_FA3_FRE" = CreateSeuratObject(D1X_FA3_FRE, meta.data = D1X_FA3_FRE.meta),
               "D4X_FA2_FRE" = CreateSeuratObject(D4X_FA2_FRE, meta.data = D4X_FA2_FRE.meta),
               "D4X_FA2_THA" = CreateSeuratObject(D4X_FA2_THA, meta.data = D4X_FA2_THA.meta),
               "D4X_FA3_FRE" = CreateSeuratObject(D4X_FA3_FRE, meta.data = D4X_FA3_FRE.meta),
               "D4X_FA3_THA" = CreateSeuratObject(D4X_FA3_THA, meta.data = D4X_FA3_THA.meta))

for (i in 1:length(sc.list)) {
  sc.list[[i]] <- NormalizeData(sc.list[[i]], verbose = FALSE)
  sc.list[[i]] <- FindVariableFeatures(sc.list[[i]], selection.method = "vst",
                                       nfeatures = 2000, verbose = FALSE)
}
sc.anchors <- FindIntegrationAnchors(object.list = sc.list, dims = 1:50)
saveRDS(sc.anchors, "./data/seurat/sc.anchors.rds")

sc.integrated <- IntegrateData(anchorset = sc.anchors, dims = 1:50)
saveRDS(sc.integrated, "./data/seurat/sc.integrated.rds")

sc.integrated = readRDS("./data/seurat/sc.integrated.rds")

write.table(t(as.matrix(sc.integrated@assays$integrated@data)), 
            file = "./data/patrick/p_c_m_n_sampled.seurat.counts.csv", quote = F, row.names = F, sep = ",")

write.table(sc.integrated@meta.data[,c("cell", "stage", "celltype", "origin")], 
            file = "./data/patrick/p_c_m_n_sampled.seurat.meta.csv", quote = F, row.names = F, sep = ",")
