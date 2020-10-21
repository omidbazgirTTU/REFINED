# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:04:34 2020

@author: obazgir
"""

from argparse import ArgumentParser

parser=ArgumentParser()

parser.add_argument('--init', type=str, default='Init.pickle')
parser.add_argument('--mapping', type=str, default='theMapping.pickle')
parser.add_argument('--evolution', type=str, default='Evolv.csv')
parser.add_argument('--num', type=int, default=5)

args= parser.parse_args()