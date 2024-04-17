# Paths
FIG1_DATA = src/runs/Section2_BNN_degeneracy/output/std_prior_preds_8_layers.json
MAX_STABLE_DATA = src/data/max_stable_sims.rds
MAX_STABLE_DATA_PREPROC = src/intermediates/max_stable_data_Gumbel.npy
MAX_STABLE_DATA_CONDSIM = src/intermediates/Max-Stable_CondSim.rds
MAX_STABLE_TRUE_FIELD = src/runs/Section5_2_SBNN-IP/output/true_field.npy
CONFIGS = $(wildcard src/configs/*.yaml)
RUNS = $(patsubst src/configs/%.yaml, src/runs/%/output/BNN_cov_matrix.npy, $(CONFIGS))
all: figures

$(FIG1_DATA): src/1_BNN_degeneracy.py
	python $<

$(MAX_STABLE_DATA): src/2a_Gen_max_stable_data.R
	Rscript $<

$(MAX_STABLE_DATA_PREPROC): src/2b_Preprocess_max_stable.py $(MAX_STABLE_DATA)
	python $<

src/runs/%/output/BNN_cov_matrix.npy: src/configs/%.yaml src/3_FitModels.py
	python src/3_FitModels.py --config $<

${MAX_STABLE_TRUE_FIELD}: src/configs/Section5_2_SBNN-IP.yaml src/3_FitModels.py
	python src/3_FitModels.py --config $<

$(MAX_STABLE_DATA_CONDSIM): src/2c_CondSim_MaxStable.R $(MAX_STABLE_DATA_PREPROC) ${MAX_STABLE_TRUE_FIELD}
	Rscript $<

figures: .figures_sentinel

.figures_sentinel: $(FIG1_DATA) $(RUNS) $(MAX_STABLE_DATA_CONDSIM)  src/4_Plot_Results.R src/utils.R
	Rscript src/4_Plot_Results.R
	touch .figures_sentinel

print:
	@echo "Configs: $(CONFIGS)"
	@echo "Targets: $(TARGETS)"
