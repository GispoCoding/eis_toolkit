"""
checks df n fieldnames for errors 
Created an Dezember 13 2022
@author: torchala 
""" 

# Stand: noch nicht begommen

# Prüfungen für Training: df und field-dictionary
# - Field-Liste entspricht vollständig der Namen der columns des df (es darf kein c-, v- oder t-Feld fehlen)
# - mindestens 1 c- oder v- columns
# - genau ein t-column muss vorhandem sein
# - v- und t-columns müssen numerisch sein 

# Prüfungen für Predition: df und flied-Dictionry
# - Es ist eine Datei myML-mdl vorhanden
# - Field-Liste entspricht vollständig der Namen der columns des df (es darf kein c-, v- oder t-Feld fehlen)
# - mindestens 1 c- oder v- columns
# - ein t-column doll nicht vorhanden sein
# - v- und t-columns müssen numerisch sein 
# - falls mindestens ei c-Feld vohanden ist, muss es eine myOhe-Datei geben

# Prüfungen für Predition nach onhotencoding: 
# - Felde in df, myML-Modell stimmen überein
