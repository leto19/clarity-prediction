def HASQI_v2(x,fs_x,y,fs_y,hearing_loss,eq,level1=65):

    xenv,xBM,yenv,yBM,xSL,ySL,fsamp = eb_EarModel(x,fs_x,y,fs_y,hearing_loss,eq,level1)
    