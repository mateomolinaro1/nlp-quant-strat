from nlp_quant_strat.data.data_loader import DataLoader, TranscriptTypes

data = DataLoader()
data.get_data(key=TranscriptTypes.UNPROCESSED.value)
