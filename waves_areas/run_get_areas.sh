python3 find_areas.py /mnt/lustre/projects/astro/general/sp624/waves_randoms/waves-wide_n_randoms.parquet --region waves-wide_n > waves_n_area_1b.txt
python3 find_areas.py /mnt/lustre/projects/astro/general/sp624/waves_randoms/waves-wide_s_randoms.parquet --region waves-wide_s > waves_s_area_1b.txt
python3 find_areas.py /mnt/lustre/projects/astro/general/sp624/waves_randoms/waves-deep_randoms.parquet --region waves-deep > waves_deep_area_1b.txt

#echo "SECOND BATCH"

#python3 find_areas.py /mnt/lustre/projects/astro/general/sp624/waves_randoms/second_batch/waves-wide_n_randoms.parquet --region waves-wide_n > waves_n_area_2.txt
#python3 find_areas.py /mnt/lustre/projects/astro/general/sp624/waves_randoms/second_batch/waves-wide_s_randoms.parquet --region waves-wide_s > waves_s_area_2.txt
#python3 find_areas.py /mnt/lustre/projects/astro/general/sp624/waves_randoms/second_batch/waves-deep_randoms.parquet --region waves-deep > waves_deep_area_2.txt

