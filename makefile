MAIN_DATA=gosuai-dota-2-game-chats/dota2_chat_messages.csv
COLUMN_DATA=column4.txt
NUM_REMOVE=600000

english.txt:
	 grep -v -P -e "[^\x00-\x7F]" $(COLUMN_DATA) | tee english.txt | wc -l

occurences.txt: english.txt
	awk -F ' ' -v 'OFS=\n' '{gsub(/[[:punct:]]/, " ")} 1' english.txt | tr "[:space:]" "\n" | sort | uniq -c | sort -n | tee occurences.txt | wc -l

rem_pattern.txt: occurences.txt
	head -n $(NUM_REMOVE) occurences.txt | awk '{print $$2}' - > rem_pattern.txt

filtered.txt: rem_pattern.txt english.txt
	awk '{a=$$0; s="\x80"; gsub(/[[:punct:]]/, " ", a); print a,s,$$0}' english.txt | tee pairs.txt | grep -a -v -F -f rem_pattern.txt - | awk -F '\x80' '{print $$2}' - | tee filtered.txt | head
