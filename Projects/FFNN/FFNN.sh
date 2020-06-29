#

log=' -log log/FFNN-cmd.log'
phones=' -phones data/phones-135.lst'

path=" -path data/"

path_f0=""
path_ph=""
#path_f0=" -path-f0 data/"
#path_ph=" -path-ph data/"

model=""
model=" -model model/FFNN-cmd-d1.pth"

list=""
#list=$list" -list data/ptdb_rand.lst"
#list=$list" -list data/aplawdw-16k.lst"
#list=$list" -list data/arctic-nali-rand.lst"

file=""
file=$file" -file arctic_a0011"
file=$file" -file arctic_a0012"

noisy=" -noisy 2"

train=""
#train=" -train"

save_f0=""
save_ph=""
save_f0=" -save-f0 temp/"
save_ph=" -save-ph temp/"

date

time python FFNN_cmd.py $train $model $list $file $log $path $path_f0 $path_ph $noisy $phones $save_f0 $save_ph

date
