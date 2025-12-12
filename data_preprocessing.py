import csv

# in_csv = "train_drivable_check.csv"
# out_csv = "train_list.csv"

# in_csv = "val_drivable_check.csv"
# out_csv = "val_list.csv"

in_csv = "test_drivable_check.csv"
out_csv = "test_list.csv"


with open(in_csv, "r", encoding="utf-8") as f_in, \
     open(out_csv, "w", newline="", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row["has_drivable"] == "1":  # only keep those
            writer.writerow(row)
print(f"Filtered CSV saved to: {out_csv}")