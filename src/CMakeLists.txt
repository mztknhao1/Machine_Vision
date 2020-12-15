aux_source_directory(./ SRC)
add_library(SRC ${SRC} myutils.cpp ../include/myutils.h ../include/BA_g2o.h)