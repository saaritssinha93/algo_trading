# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:20:12 2025

@author: Saarit
"""

import os

# List of tickers
tickers = [
    "ACL", "ADVANIHOTR", "AGIIL", "ALICON", "APCOTEXIND", "BAJAJHCARE",
    "BALAJEE", "BANSWRAS", "BCONCEPTS", "BIRLACABLE", "BSHSL", "BUTTERFLY",
    "CARYSIL", "CENTENKA", "CENTRUM", "CENTUM", "CHEMBOND", "CHEVIOT",
    "CONSOFINVT", "CSLFINANCE", "DECCANCE", "DEEDEV", "DHUNINV", "DPWIRES",
    "DYCL", "ELDEHSG", "EMAMIPAP", "ENIL", "EXCELINDUS", "FAIRCHEMOR",
    "GEECEE", "GENUSPAPER", "GFLLIMITED", "HINDCOMPOS", "HMAAGRO", "HMVL",
    "HTMEDIA", "IFBAGRO", "IFGLEXPOR", "IMPAL", "INDOBORAX", "INDORAMA",
    "INTLCONV", "IRISDOREME", "JAICORPLTD", "JAYAGROGN", "JAYBARMARU",
    "JINDALPHOT", "JISLDVREQS", "JNKINDIA", "KANORICHEM", "KHADIM", "KILITCH",
    "KOKUYOCMLN", "KRISHANA", "LINC", "MAGADSUGAR", "MANAKSIA", "MATRIMONY",
    "MEDICAMEQ", "MICEL", "MIRZAINT", "MUNJALSHOW", "MUTHOOTCAP", "NAHARCAP",
    "NAHARINDUS", "NINSYS", "OMAXE", "ORIENTCER", "ORIENTTECH", "PTL",
    "PVSL", "PYRAMID", "REPRO", "RML", "RUBYMILLS", "SAKAR", "SANDESH",
    "SANGAMIND", "SAURASHCEM", "SESHAPAPER", "SHREYANIND", "SINCLAIR",
    "SPECIALITY", "SREEL", "SUKHJITS", "SUTLEJTEX", "SUYOG", "TEXINFRA",
    "TRF", "TTKHLTCARE", "UNIENTER", "UYFINCORP", "VAKRANGEE", "VALIANTLAB",
    "VERANDA", "VHL", "VINYLINDIA", "YUKEN"
]

# Directory where the CSV files are located
directory = "main_indicators2"

for ticker in tickers:
    filename = f"{ticker}_main_indicators.csv"
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Removed {filepath}")
        except Exception as e:
            print(f"Error removing {filepath}: {e}")
    else:
        print(f"File not found: {filepath}")
