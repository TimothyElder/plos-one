## Code associated with extracting the subjects from the plos articles 

file_names = []
raw_contents = []
for file in (file for file in os.scandir(target_dir) if file.is_file() and not file.name.startswith('.')):
    file_names.append(file.name)
    with open(file.path, encoding="utf8") as f:
        raw_contents.append(BeautifulSoup(f.read(), 'xml'))
        


temp_subjects = []
for i in raw_contents:
    temp_subjects.append(i.findAll('subject'))



set_subjects = []
subjects_temp = []
for i in temp_subjects:
    set_subjects=set(i)
    subjects_temp.append(list(set_subjects))


subjects = []
for nlist in subjects_temp:
    row = []
    for item in nlist:
       row.append(item.text)
    subjects.append(row)



subjects_normal = []
for nlist in subjects:
    row = []
    for item in nlist:
       row.append(item.lower())
    subjects_normal.append(row)