# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:30:24 2019

@author: p000526841
"""
import pandas as pd

# ログのライブラリ
import logging
from logging import getLogger, StreamHandler, Formatter, FileHandler

class LogParameters:
    def __init__(self):

        self.logger_name_ = None
        self.logger_ = None
        


class MyLogger:
    def __init__(self):
        self.loggers_ = {}
    
    def __del__(self):
        
        for each_logger in self.loggers_.values():
            each_logger.handlers= []

    def generateLogger(self, _logger_name, _path_to_log="../data/log/debug_log2.log"):

        logger = None
        
        if self.loggers_.get(_logger_name, 0) != 0:
    
            logger = self.loggers_[_logger_name]
            logger.debug("Find logger : " + _logger_name)
            
    
        else:
            
                     
            
            # --------------------------------
            # 1.loggerの設定
            # --------------------------------
            # loggerオブジェクトの宣言
            

            logger = getLogger(_logger_name)
        
            # loggerのログレベル設定(ハンドラに渡すエラーメッセージのレベル)
            # ERRORを設定したためDEBUGは表示されない
            logger.setLevel(logging.DEBUG)
    
            # --------------------------------
            # 2.handlerの設定
            # --------------------------------
            # handlerの生成
            stream_handler = StreamHandler()
            
            # handlerのログレベル設定(ハンドラが出力するエラーメッセージのレベル)
            stream_handler.setLevel(logging.DEBUG)
        
            # ログ出力フォーマット設定
            #handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            #stream_handler.setFormatter(handler_format)
            
            # ---- 2-2.テキスト出力のhandler ----
            # handlerの生成
            file_handler = FileHandler(_path_to_log, 'a', encoding='utf-8')
            
            # handlerのログレベル設定(ハンドラが出力するエラーメッセージのレベル)
            file_handler.setLevel(logging.DEBUG)
            
            # ログ出力フォーマット設定
            #file_handler.setFormatter(handler_format)
        
            # --------------------------------
            # 3.loggerにhandlerをセット
            # --------------------------------
            logger.addHandler(stream_handler)
            
            # テキスト出力のhandlerをセット
            logger.addHandler(file_handler)
            
        
            self.loggers_[_logger_name] = logger
            logger.debug("Initial logger : " + _logger_name)
            
        return logger
                 
#if __name__ == '__main__':
#    
#    df_train = pd.read_csv("../data/raw/train.csv", index_col=0)
#    
#    my_logger = MyLogger()
#    logger = my_logger.generateLogger("EDA", "../data/log/test_debug.log")
#    
#    logger.debug(df_train.loc[df_train["production_countries"].isnull() == True, ["spoken_languages", "original_language"]])
