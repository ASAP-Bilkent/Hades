.PHONY: help train-bco train-bcd train-mnist poseidon exp% test% test clean latex%_graph latex%_table latex_pca
.NOTPARALLEL: test 

help:
	@echo "Available commands:"
	@echo "  make train-bco      # Train on Breast Cancer Original dataset"
	@echo "  make train-bcd      # Train on Breast Cancer Diagnostic dataset"
	@echo "  make train-mnist    # Train on MNIST dataset"
	@echo "  make exp<N>         # Run experiment N (e.g. make exp1)"
	@echo "  make latex<N>       # Generate table for experiment N (e.g. make latex1)"
	@echo "  make latex<N>_graph # Generate graph for experiment N (e.g. make latex1_graph)"
	@echo "  make test           # Run all HE tests"
	@echo "  make test<N>        # Run test N (e.g. make test2)"
	@echo "  make clean          # Clean temporary files"

train-bco:
	python3 main.py --dims=4 --batch_size=1 --dataset=bco --learning_rate=0.1 --no_enc --no_idlg --data_amount=-1 --epochs=1 --act_func=sigmoid

train-bcd:
	python3 main.py --dims=16 --batch_size=1 --dataset=bcd --learning_rate=0.1 --no_enc --no_idlg --data_amount=-1 --epochs=1 --act_func=sigmoid

train-mnist:
	python3 main.py --dims=128,16 --batch_size=1 --dataset=mnist --learning_rate=0.1 --no_enc --no_idlg --data_amount=-1 --epochs=1 --act_func=sigmoid

poseidon:
	python3 main.py --n_synth_features=4 --dims=4 --batch_size=1 --dataset=synth --no_idlg --no_infer --data_amount=1 --epochs=1

exp%:
	python3 main.py --exp=$*

latex%_graph:
	python3 src/experiments/exp$*_graph.py

latex%:
	python3 src/experiments/exp$*_table.py

latex_pca:
	python3 src/experiments/pca_exp.py

test%:
	python3 main.py --test=$*

test: test1 test2 test3 test4 test5 test6

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete