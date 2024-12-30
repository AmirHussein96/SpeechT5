fmpeg -version
conda install -c conda-forge ffmpeg=6

# python ../../examples/wav2vec/wav2vec_manifest.py /mm2/scratch/ahussein/libri100/train/ --valid-percent 0 --dest manifests

tsv_dir=/scratch4/skhudan1/ahussein/SpeechT5
split=train_libri100
ckpt_path=/scratch4/skhudan1/ahussein/hubert_base_ls960.pt
layer=9
nshard=1
rank=0
feat_dir=/scratch4/skhudan1/ahussein/SpeechT5/hubert_feat

python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

km_path=/scratch4/skhudan1/ahussein/SpeechT5/kmean500
n_cluster=500 #1024  # number of clusters
python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1

lab_dir=/scratch4/skhudan1/ahussein/SpeechT5/kmeans_lab500

python fairseq/examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}


for x in $(seq 0 $((n_cluster - 1))); do
  echo "$x 1"
done >> $lab_dir/dict.km.txt
