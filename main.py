from nlp_quant_strat.data.data_loader_old import DataLoader, TranscriptTypes

data = DataLoader()
data.get_data(key=TranscriptTypes.UNPROCESSED.value)
