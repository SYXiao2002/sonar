from sonar.utils.file_helper import save_head_csv

save_head_csv(
	input_path='res/traningcamp-nirspark/HC1-hbo_converted.csv',
	output_path='res/traningcamp-nirspark/test-debug.csv',
	start_row=1,
	end_row=11*1000
)
