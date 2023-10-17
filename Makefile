# Paths
FIG1_DATA = src/runs/Section2_BNN_degeneracy/output/std_prior_preds_8_layers.json
CONFIGS = $(wildcard src/configs/*.yaml)
TARGETS = $(patsubst src/configs/%.yaml, src/runs/%/output/BNN_cov_matrix.npy, $(CONFIGS))
all: $(FIG1_DATA) $(TARGETS) figures

$(FIG1_DATA): src/BNN_degeneracy.py
	python $<

src/runs/%/output/BNN_cov_matrix.npy: src/configs/%.yaml src/FitModels.py
	python src/FitModels.py --config $<

figures: .figures_sentinel

.figures_sentinel: $(TARGETS) $(FIG1_DATA) src/Plot_Results.R src/utils.R
	Rscript src/Plot_Results.R
	touch .figures_sentinel

print:
	@echo "Configs: $(CONFIGS)"
	@echo "Targets: $(TARGETS)"
