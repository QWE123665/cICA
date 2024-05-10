library(SeuratObject)
library(Matrix)
library(readr)
library(Seurat)


human=readRDS('/Users/kexinwang/Documents/harvard/research/contrastive_learning/python/monkey_human_data/human_SCT_UMI_expression_matrix.RDS')
human_less=subset(x = human, downsample = 10000)
human_less_data_normalized=NormalizeData(object=human_less)
saveRDS(human_less_data_normalized, file = "human_less_data_normalized", ascii = FALSE, version = NULL,
        compress = TRUE, refhook = NULL)
human=readRDS('/Users/kexinwang/Documents/harvard/research/contrastive_learning/python/human_less_data_normalized', refhook = NULL)
human_data=as.data.frame(human@assays$RNA@data)
write.csv(human_data, "human_data.csv")


gorilla=readRDS('/Users/kexinwang/Documents/harvard/research/contrastive_learning/python/monkey_human_data/gorilla_SCT_UMI_expression_matrix.RDS')
gorilla_less=subset(x = gorilla, downsample = 10000)
gorilla_less_data_normalized=NormalizeData(object=gorilla_less)
saveRDS(gorilla_less_data_normalized, file = "gorilla_less_data_normalized", ascii = FALSE, version = NULL,
        compress = TRUE, refhook = NULL)
gorilla=readRDS('gorilla_less_data_normalized', refhook = NULL)
gorilla_data=as.data.frame(gorilla@assays$RNA@data)
write.csv(gorilla_data, "gorilla_data.csv")

chimp=readRDS('/Users/kexinwang/Documents/harvard/research/contrastive_learning/python/monkey_human_data/chimp_SCT_UMI_expression_matrix.RDS')
chimp_less=subset(x = chimp, downsample = 10000)
chimp_less_data_normalized=NormalizeData(object=chimp_less)
saveRDS(chimp_less_data_normalized, file = "chimp_less_data_normalized", ascii = FALSE, version = NULL,
        compress = TRUE, refhook = NULL)
chimp=readRDS('/Users/kexinwang/Documents/harvard/research/contrastive_learning/python/chimp_less_data_normalized', refhook = NULL)
chimp_data=as.data.frame(chimp@assays$RNA@data)
write.csv(chimp_data, "chimp_data.csv")
