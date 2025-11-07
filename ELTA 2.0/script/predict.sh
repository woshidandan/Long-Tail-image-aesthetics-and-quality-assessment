# #sh script/predict.sh
# python main.py -e                   /home/llm/dataset_csv/tad66k/test.csv \
#                --test_dataset_path  /data/dataset/TAD66K \
#                --resume             /home/llm/elta/code/checkpoint/0425_1535/ckpt.pth

python main.py --csv_path           /home/llm/dataset_csv/tad66k \
               --dataset_path       /data/dataset/TAD66K