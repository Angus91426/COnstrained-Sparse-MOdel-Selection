# Export the riboflavin data of Buhlmann, Kalisch & Meier (2014) into the CSV files used by
# the code (alternative to Data/prepare_riboflavin.py). From the repository root, run:
#
#     Rscript Data/prepare_riboflavin.R
#
# The R package hdi, which distributed these data, was archived on CRAN on 2026-07-10, so
# this script does not install it: it downloads the archived source package hdi_0.1-9.tar.gz
# from the CRAN archive and loads the riboflavin data file from it.
#
# Writes Data/Riboflavin/riboflavin_x.csv (71 x 4088 log expression levels) and
# Data/Riboflavin/riboflavin_y.csv (log riboflavin production rate), both keyed by `sample`.
# R writes numbers with 15 significant digits, which Data/check_real_data.py accepts.

url <- "https://cran.r-project.org/src/contrib/Archive/hdi/hdi_0.1-9.tar.gz"
tmp <- tempfile()
dir.create(tmp)
tarball <- file.path(tmp, "hdi_0.1-9.tar.gz")
download.file(url, tarball, mode = "wb")
untar(tarball, exdir = tmp)
data_file <- list.files(file.path(tmp, "hdi", "data"), pattern = "^riboflavin\\.", full.names = TRUE)[1]
load(data_file)

x <- unclass(riboflavin$x)
samples <- rownames(riboflavin)

dir.create(file.path("Data", "Riboflavin"), showWarnings = FALSE, recursive = TRUE)
write.csv(data.frame(sample = samples, x, check.names = FALSE),
          file.path("Data", "Riboflavin", "riboflavin_x.csv"), row.names = FALSE)
write.csv(data.frame(sample = samples, y = as.numeric(riboflavin$y)),
          file.path("Data", "Riboflavin", "riboflavin_y.csv"), row.names = FALSE)
cat("Wrote riboflavin_x.csv and riboflavin_y.csv to Data/Riboflavin\n")
