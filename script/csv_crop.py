from sonar.utils.file_helper import save_head_csv

save_head_csv(
	input_path='res/trainingcamp-no-filter-test/hbo/HC1.csv',
	output_path='res/trainingcamp-no-filter-test/hbo/TEST-1.csv',
	start_row=1,
	end_row=11*500
)
