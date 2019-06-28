#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @author   : Feifei
# @IDE      : Pycharm
# @file     : deepcopy.py
# @time     : 2019/5/31 10:50

from copy import deepcopy

hello_string = ['hEllo','woRld','!']
c_hello_string = deepcopy(hello_string)
del hello_string[-1]
l_hello = [h.replace('l','Y') for h in c_hello_string if len(h)==5]
t=list(map(lambda  c:str(c).capitalize(),l_hello))
print(t)




