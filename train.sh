################################################################## Dynamic Object (FastGS)
# FastGS is enabled by default. For a baseline comparison, append --disable_fastgs to the train_gui.py command.
# output=output/dynamic_object_fastgs
# exp=fan
# python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynObjects/data/fan -m $output/$exp --max_time 0.70 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 32 --vel_start_time 0.0
# python seg.py -m $output/$exp --K 2 --vis
# python render.py -m $output/$exp --mode render --skip_train
# python metrics.py -m $output/$exp --s test --half_res
# python metrics.py -m $output/$exp --s val --half_res

# exp=whale
# python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynObjects/data/whale -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python render.py -m $output/$exp --mode render --skip_train
# python metrics.py -m $output/$exp --s test --half_res
# python metrics.py -m $output/$exp --s val --half_res

# exp=shark
# python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynObjects/data/shark -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python render.py -m $output/$exp --mode render --skip_train
# python metrics.py -m $output/$exp --s test --half_res
# python metrics.py -m $output/$exp --s val --half_res

# exp=telescope
# python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynObjects/data/telescope -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python render.py -m $output/$exp --mode render --skip_train
# python metrics.py -m $output/$exp --s test --half_res
# python metrics.py -m $output/$exp --s val --half_res

# exp=fallingball
# python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynObjects/data/fallingball -m $output/$exp --max_time 0.72 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python render.py -m $output/$exp --mode render --skip_train
# python metrics.py -m $output/$exp --s test --half_res
# python metrics.py -m $output/$exp --s val --half_res

# exp=bat
# python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynObjects/data/bat -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# # python seg.py -m $output/$exp --K 3 --vis
# python render.py -m $output/$exp --mode render --skip_train
# python metrics.py -m $output/$exp --s test --half_res
# python metrics.py -m $output/$exp --s val --half_res
# python render.py -m $output/$exp --mode all --skip_val --skip_test


output=output/dynamic_indoor_fastgs

exp=chessboard
python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynIndoorScene/data/chessboard -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python seg.py -m $output/$exp --K 3 --vis
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res

exp=darkroom
python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynIndoorScene/data/darkroom -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python seg.py -m $output/$exp --K 3 --vis
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res

exp=dining
python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynIndoorScene/data/dining -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python seg.py -m $output/$exp --K 3 --vis
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res

exp=factory
python train_gui.py -s /data/lunengbo/LNB/VeloGauss/data/DynIndoorScene/data/factory -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python seg.py -m $output/$exp --K 3 --vis
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res