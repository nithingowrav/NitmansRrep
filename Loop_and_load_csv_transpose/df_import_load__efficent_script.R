getwd()

#Point working directory to my CSV Source
setwd("C:/Users/Nithin Gowrav/Documents/R_Source/")

#create an empty list to hold the files
df_src<-data.frame()

#put all the csv files in the directory into a vector
csv_file_list <- dir(pattern = "*.csv")
month<-c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
attr<-c("sls_avg_prc","rtl_avg_prc","tot_arvl")

df<-data.frame()

#create a loop to load all the csv files mentioned in the vector - skip option is used as the header is messed up when the file is converted to csv
for (i in 1:length(csv_file_list)){
  df_src <- read.csv(csv_file_list[i])
    year<-df_src[2,1]
    df_sls_avg_prc<-setNames(data.frame(df_src[9,seq(3,36, length=12)],check.names = F),month)
    df_rtl_avg_prc<-setNames(data.frame(df_src[9,seq(4,37, length=12)],check.names = FALSE),month)
    df_tot_arrvl<-setNames(data.frame(df_src[9,seq(5,38, length=12)],check.names = FALSE),month)
    df_stg<-rbind(df_sls_avg_prc,df_rtl_avg_prc,df_tot_arrvl)
    rownames(df_stg)=attr
    trnps_df_stg<-data.frame(t(df_stg))
    df<-rbind(df,cbind(year,trnps_df_stg))
    tgt_final<-data.frame(cbind(month,df), row.names = NULL)
}
