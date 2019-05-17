import quandl
dow_code = 'BCB/UDJIAD1'
quandl.get(dow_code)
quandl.export_table()
print(quandl)