selected_stocks = {
    "ABSLBANETF",   # Aditya Birla – Banking & Fin Services ETF (Nifty Fin 25/50)
    "ABSLNN50ET",   # Aditya Birla – Nifty Next 50 ETF
    "ABSLPSE",      # Aditya Birla – PSU Enterprise thematic basket
    "AUTOIETF",     # Motilal Oswal – Nifty Auto sector ETF
    "AXISBPSETF",   # Axis AMC – Bharat PSU Equity basket (CPSE-style)
    "AXISCETF",     # Axis – Consumption & FMCG thematic ETF
    "AXISILVER",    # Axis – Physical Silver ETF
    "BANKBEES",     # Nippon India – Nifty Bank ETF
    "BBNPPGOLD",    # Nippon India – Physical gold ETF (BeES format)
    "BBNPNBETF",    # Bharat-Bond (AAA PSU ladder, Apr-33 maturity)
    "BFSI",         # Motilal Oswal – Nifty Financial Services 25/50 ETF
    "BSE500IETF",   # Motilal Oswal – S&P BSE 500 broad-market ETF
    "BSLGOLDETF",   # Aditya Birla – Physical Gold ETF
    "BSLSENETFG",   # Aditya Birla – BSE Sensex ETF
    "COMMOIETF",    # Invesco – Broad commodity futures basket
    "CONSUMBEES",   # Nippon India – Nifty Consumption ETF
    "CPSEETF",      # CPSE ETF (10 listed PSU firms – Govt divestment vehicle)
    "DIVOPPBEES",   # Nippon – Dividend Opportunities high-yield basket
    "EQUAL50ADD",   # Edelweiss – Nifty 50 Equal-Weight ETF (add-series)
    "EGOLD",        # Kotak – Physical Gold ETF
    "FINIETF",      # ICICI Pru – Nifty Financial Services ETF
    "GOLDBEES",     # Nippon – Physical Gold (BeES) ETF
    "HDFCGROWTH",   # HDFC – Nifty Growth factor ETF
    "HDFCNIFBAN",   # HDFC – Nifty Bank ETF
    "HDFCSML250",   # HDFC – Nifty Smallcap 250 ETF
    "HDFCVALUE",    # HDFC – Nifty 50 Value 20 factor ETF
    "HEALTHIETF",   # ICICI Pru – Nifty Healthcare ETF
    "HEALTHY",      # Motilal Oswal – Healthcare momentum basket
    "INFRAIETF",    # ICICI Pru – Nifty Infrastructure ETF
    "ITBEES",       # Nippon – Nifty IT sector ETF
    "IVZINNIFTY",   # Invesco – Nifty 50 tracker (high-NAV units)
    "JUNIORBEES",   # Nippon – Nifty Next 50 ETF
    "LICNETFN50",   # LIC MF – Nifty 50 ETF
    "LICNETFGSC",   # LIC MF – Nifty G-Sec 10 yr constant ETF
    "LICNETFSEN",   # LIC MF – Sensex ETF
    "LICNMID100",   # LIC MF – Nifty Midcap 100 ETF
    "LOWVOL",       # ICICI Pru – Nifty Low Volatility 30 ETF
    "LOWVOLIETF",   # Invesco – Nifty Low Volatility 30 ETF
    "MAFANG",       # Mirae – FANG/Tech innovators thematic ETF
    "MAKEINDIA",    # Motilal Oswal – Make-in-India manufacturing basket
    "MASPTOP50",    # Motilal Oswal – Alpha Top 50 equal-weight ETF
    "MID150BEES",   # Nippon – Nifty Midcap 150 ETF
    "MIDCAP",       # Motilal Oswal – S&P BSE MidCap ETF
    "MIDQ50ADD",    # Edelweiss – Nifty Midcap Quality 50 add-series
    "MIDSMALL",     # Motilal – Nifty MidSmallcap 400 ETF
    "MIDSELIETF",   # ICICI – Nifty MidSmallcap 400 ETF
    "MOM100",       # Motilal – Momentum 100 factor ETF
    "MOM30IETF",    # ICICI – Nifty 200 Momentum 30 ETF
    "MOM50",        # Motilal – Momentum 50 blend ETF
    "MOMOMENTUM",   # Mirae – Nifty Momentum style ETF
    "MULTICAP",     # ICICI – Nifty 500 Multicap 50:25:25 ETF
    "NIF100BEES",   # Nippon – Nifty 100 ETF
    "NIF5GETF",     # Nippon – Nifty top-5 equal-weight micro ETF
    "NIF10GETF",    # Nippon – Nifty top-10 equal-weight micro ETF
    "NIFTY1",       # Motilal – Nifty 50 micro-lot ETF
    "NIFTYBEES",    # Nippon – Nifty 50 BeES ETF
    "NIFTYQLITY",   # ICICI – Nifty Quality 30 ETF
    "PHARMABEES",   # Nippon – Nifty Pharma ETF
    "PVTBANIETF",   # Edelweiss – Nifty Private Bank ETF
    "PVTBANKADD",   # Edelweiss – dividend-reinvest add line for same
    "PSUBANK",      # Kotak – Nifty PSU Bank ETF
    "PSUBNKBEES",   # Nippon – Nifty PSU Bank BeES
    "QGOLDHALF",    # Nippon – Gold half-gram micro ETF
    "QNIFTY",       # Quantum – Nifty 50 Shariah compliant ETF
    "SBIETFCON",    # SBI – Consumption index ETF
    "SBIETFPB",     # SBI – PSU Bank equal-weight mini ETF
    "SBIETFIT",     # SBI – IT sector mini ETF
    "SBIBPB",       # SBI – PSU Bond Plus SDL index ETF
    "SBINEQWETF",   # SBI – Nifty Quality 30 clone
    "SDL26BEES",    # Nippon – Target-maturity SDL Apr-2026 ETF
    "SETFNIF50",    # SBI – Nifty 50 ETF
    "SETFNIFBK",    # SBI – Nifty Bank ETF
    "SETFGOLD",     # SBI – Physical Gold ETF
    "SENSEXADD",    # Nippon – Sensex add-series
    "SENSEXETF",    # Nippon – Sensex ETF
    "SHARIABEES",   # Nippon – Shariah Nifty 50 ETF
    "SILVER",       # ICICI – Physical Silver ETF (high-NAV)
    "SILVERBEES",   # Nippon – Physical Silver BeES ETF
    "TATSILV",      # TATA AMC – Physical Silver ETF
    "TOP100CASE",   # Motilal – Nifty 100 equal-weight smart-beta
    "UTINIFTETF",   # UTI – Nifty 50 ETF
    "UTINEXT50",    # UTI – Nifty Next 50 ETF
    "UTISENSETF",   # UTI – Sensex ETF
    "UTISXN50",     # UTI – *Sensex Next 50* thematic ETF
    "HNGSNGBEES",   # Mirae – Hang-Seng Tech feeder ETF
    "ICICIB22",     # ICICI – CPSE Bharat-22 style equity basket
    "SETFSN50",     # SBI – Nifty Next 50 equal-weight variant
    "SETFBSE100",   # SBI – BSE 100 equal-weight ETF
    "MON100",       # Motilal Oswal – NASDAQ-100 (alt symbol to N100)
    "MONIFTY500",   # Motilal Oswal – Nifty-500 broad-market ETF
    "NIFTYETF",     # Mirae Asset – low-cost Nifty-50 tracker
    "NIFTYBETF",    # Bajaj Finserv – Nifty-50 tracker
    "NIFTY100EW",   # Kotak – Nifty-100 Equal-Weight ETF
    "EVINDIA",      # Mirae – Nifty EV & New-Age Auto theme
    "EVIETF",       # ICICI – same EV basket (alt AMC)
    "GROWWEV",      # Groww – same EV basket (third AMC)
    "MODEFENCE",    # Motilal – Nifty India Defence ETF
    "MOREALTY",     # Motilal – Nifty Realty ETF
    "AUTOBEES",     # Nippon – Nifty Auto (duplicate of AUTOIETF family)
    "HDFCQUAL",     # HDFC – Nifty 100 Quality-30 ETF
    "BANKNIFTY1",   # Kotak – Nifty Bank (trades as BANKNIFTY1)
    "GOLD1",        # Kotak – Physical Gold ETF
    "LIQUIDBEES",   # Nippon – collateral cash / liquid ETF
     "ALPHAETF"      # Edelweiss – Alpha Low-Risk Factor ETF
}
 