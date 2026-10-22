# Notatka zespołu liderskiego — Warsztat 2 (pełna)

**Przedmiot:** Sztuczna inteligencja w grach i rozszerzonej rzeczywistości (WFiIS AGH)  
**Prowadzący przedmiotu:** dr Piotr A. Kowalski  
**Temat:** Agent uczący się przejścia labiryntu — **tablicowy Q-learning**  
**Dla kogo jest ten plik:** zespół liderski / przewodnicy / przygotowanie zajęć  
**Nie oddawać studentom jako „ściągę z odpowiedziami”** (studenci dostają szkielet + szablon sprawozdania)

---

## 0. W jednym akapicie — o co chodzi

Studenci dostają **gotowy silnik labiryntu** i uzupełniają 4 bloki Q-learningu. Agent ma nauczyć się dochodzić do celu lepiej niż agent losowy.  
Warsztat uczy intuicji RL: **nagroda**, **eksploracja (ε)**, **oddzielenie treningu od testu** oraz ograniczenie metody (**jedna plansza = jedna tabela Q**).  
Dla nas: pakiet jest gotowy (`.py` ≈ `.ipynb`), wzorzec działa, sprawozdanie wymaga samodzielnego odczytu parametrów z kodu.

---

## 1. Temat — co znaczą pojęcia

| Pojęcie | Znaczenie u nas |
| --- | --- |
| **Agent** | Program wybierający ruch na podstawie stanu (pozycja) i tabeli Q / losowości |
| **Stan** | Pole `(wiersz, kolumna)` |
| **Akcja** | 0↑ 1↓ 2← 3→ |
| **Nagroda** | Sygnał po ruchu: cel / krok / ściana |
| **Epizod** | Jedna próba: start → cel albo limit kroków |
| **Q-learning** | Uczenie ze wzmocnieniem: aktualizacja ocen par stan–akcja |
| **Tablicowy** | Wiedza w `numpy` array, bez sieci neuronowych / GPU |
| **ε-zachłanny** | Z pr. ε ruch losowy, inaczej najlepsze Q (remisy losowane) |

### Analogie do tłumaczenia grupie
- **Szczur w labiryncie** — kara za ścianę, nagroda za wyjście, pamięć po próbach.  
- **Nauka gry punktami** — nikt nie podaje trasy; agent odkrywa ją przez nagradzanie.  
- **Notatnik ocen strzałek** — na każdym polu 4 liczby; wyższa = chętniej wybierana.

---

## 2. Cele warsztatu

### Cel praktyczny (co mają zrobić)
Nauczyć agenta dojścia do wyjścia i **porównać** go z agentem losowym na tej samej liście startów testowych.

### Cel dydaktyczny (czego mają się nauczyć)
1. Rola **projektu nagród** (co agent naprawdę optymalizuje).  
2. Rola **eksploracji** — dlaczego w treningu bywają „gorsze” ruchy.  
3. **Trening ≠ test** — w teście ε=0 i bez update Q.  
4. Ograniczenie: wynik dotyczy **tej mapy**, nie „wszystkich labiryntów”.

### Czego NIE jest to warsztat
Nie budujemy gry AAA, nie trenujemy deep RL, nie wymagamy GPU.

---

## 3. Jak wygląda środowisko (zasady „gry”)

### Oznaczenia `BOARD`
- `0` — wolne  
- `1` — ściana  
- `2` — cel (dokładnie jedno pole)

### Nagrody (`step`)
| Sytuacja | Nagroda | Pozycja |
| --- | --- | --- |
| Wejście na cel | **+10** | na celu, sukces |
| Zwykły ruch | **−0,1** | nowe pole |
| Ściana / poza mapą | **−1** | **bez zmiany** pozycji |

### Koniec epizodu
Cel **lub** `MAX_STEPS = 50` (wtedy porażka).

### Dlaczego −0,1 za ruch?
Żeby preferować **krótsze** trasy, nie tylko „dojście kiedyś”.

### Parametry bazowe (zakres podstawowy)
| Parametr | Wartość |
| --- | --- |
| Epizody | 300 |
| α | 0,1 |
| γ | 0,95 |
| ε trening | 0,2 |
| ε test | 0,0 |
| Limit kroków | 50 |
| Starty testowe | 20 (`TEST_STARTS`, seed 42) |
| Seed | 42 |

---

## 4. Przebieg zajęć (dla nas)

| Blok | Czas | Co się dzieje |
| --- | --- | --- |
| Prezentacja liderska | ~10 min | Cel, zasady, harmonogram, oddanie |
| Praca zespołów + przewodnicy | 90 min | TODO + eksperyment + sprawozdanie |
| Dema | 5–6 min / zespół | Pokaz + liczby + wniosek |
| Etap III | po zajęciach | Krótkie podsumowanie liderskie na Teams |

### Harmonogram 90 min (przewodnik)
| Minuty | Fokus |
| --- | --- |
| 0–10 | Demo losowe, otwarcie sprawozdania, role |
| 10–25 | TODO 1–2 (Q + ε-greedy + remisy) |
| 25–55 | TODO 3 + trening + wykres |
| 55–75 | TODO 4 + porównanie (+ opcjonalnie ε) |
| 75–90 | Ścieżka, wnioski, oddanie |

### Rola przewodnika
Pomaga, naprowadza, **nie koduje za zespół**. Jakość pomocy jest oceniana.

---

## 5. Co jest w folderze (mapa plików)

### Dla zespołów wykonawczych (pakiet startowy)
| Plik | Rola |
| --- | --- |
| `warsztat2_szkielet.py` | Szkielet TODO 1–4 (skrypt) |
| `warsztat2_szkielet.ipynb` | **Ten sam kod i komentarze**, w komórkach |
| `README.md` | Instalacja / uruchomienie / oddanie |
| `requirements.txt` | `numpy==1.26.4`, `matplotlib==3.8.4` |
| `requirements-notebook.txt` | + Jupyter (opcjonalnie) |
| `Szablon_sprawozdania.md` | Sprawozdanie — **bez gotowych liczb**, trzeba odczytać z kodu |

### Dla nas (wewnętrzne)
| Plik | Rola |
| --- | --- |
| `warsztat2_rozwiazanie_wzorcowe.py` | Pełne TODO + te same komentarze szkieletu; artefakty `*_ref*` / `*_referencyjna*` |
| `plan.md`, `plan_etap1.md`, `plan_etap2.md` | Konspekt i checklisty |
| `Instrukcja_warsztatow.pdf` | Oryginał instrukcji zajęć |
| `NOTATKA_WARSZTAT2.md` | **Ta notatka** |
| PNG / NPY referencyjne | Przykładowe wyniki po uruchomieniu wzorca |

### Stan plansz w kodzie (ważne!)
- **Szkielet (`.py` / `.ipynb`):** domyślnie **5×5** (zakres podstawowy dla studentów).  
- **Wzorzec:** obecnie ustawiony na **7×7** (5×5 zakomentowane) — do dema / weryfikacji większej mapy.  
- Po zmianie `BOARD` i restarcie: `VALID_STARTS`, `TEST_STARTS`, kształt Q aktualizują się same.

---

## 6. Co studenci uzupełniają (TODO) — i jak wygląda rozwiązanie

Miejsca w szkielecie = `NotImplementedError` / `# UZUPEŁNIJ TUTAJ`.  
We wzorcu = te same docstringi + znaczniki **`UZUPEŁNIONE (TODO N)`**.

### TODO 1 — `create_q_table`
**Oczekiwane:** `np.zeros((*BOARD.shape, 4))`  
**Sens:** na start agent „nic nie wie”.

### TODO 2 — `choose_action`
**Oczekiwane:**
- z pr. ε → losowa akcja 0..3  
- inaczej → max Q; **remisy losować** (`np.where` / `np.isclose` + `choice`)  
**Pułapka:** zwykły `argmax` przy samych zerach zawsze bierze akcję 0.

### TODO 3 — aktualizacja w `train_q_learning`
**Oczekiwane:**
- jeśli cel: `target = reward` (przyszłość = 0)  
- inaczej: `target = reward + gamma * max Q[next]`  
- `Q[s,a] += alpha * (target - Q[s,a])`  
**Pułapka:** branie `max Q` z pola celu jako kontynuacji.

### TODO 4 — `evaluate_agent`
**Oczekiwane:** pętla po `test_starts`; wyuczony: ε=0, **bez** update Q; losowy: `randint`; zwrócić m.in. `success_rate`, `success_count`, `avg_steps_success`, `results`.  
**Pułapka:** ε>0 w teście albo dalsze uczenie; pomijanie porażek w %.

---

## 7. Zakresy zadania

### A) Podstawowy (priorytet)
1. TODO 1–4  
2. Trening 300 / α=0,1 / γ=0,95 / ε=0,2  
3. Test: losowy vs wyuczony na 20 startach  
4. Wykres nagród + przykładowa ścieżka  
5. Sprawozdanie (odczyt danych z kodu)

### B) Dodatkowy — porównanie ε
Trening dla ε=0,05 i ε=0,30 (reszta parametrów ta sama).  
Funkcja w kodzie: `run_extension_experiment()`.

### C) Dodatkowy — większa plansza (sekcja 6 sprawozdania)
Dopiero gdy A działa i artefakty są OK:
- zamiana `BOARD` np. na 7×7  
- osobne nazwy plików (nie nadpisać 5×5)  
- ewentualnie więcej epizodów / inny ε / `MAX_STEPS`  
- porównanie skuteczności i ścieżek

### D) Podpowiedź badawcza — ściana `(4, 2)` na 7×7 (sekcja 6.4)
Ustawić `BOARD[4, 2] = 1`.  
Często **min. liczba ruchów się nie zmienia**, a **wyuczona ścieżka jest inna**.

**Dlaczego (wniosek dla nas / do naprowadzania):**
- wiele tras o tej samej długości,
- Q-learning wzmacnia **ćwiczone** przejścia, nie „oficjalne BFS”,
- ściana zmienia graf i historię epizodów (nawet przy tym samym seedzie mapa jest inna),
- ε + remisy Q → inna polityka końcowa.

**Wpływ na metryki:** rysunek — często mocno; % sukcesu / śr. kroki — mogą zostać podobne albo lekko się zmienić.

---

## 8. Wyniki referencyjne (dla nas — orientacja)

Poniższe liczby z rozwiązania wzorcowego (`SEED=42`, 300 ep., α=0,1, γ=0,95).  
Lekkie różnice między uruchomieniami są możliwe przy zmianie kolejności RNG, ale rząd wielkości powinien się zgadzać.

### 8.1. Plansza 5×5 (jak w szkielecie studentów)

```text
BOARD 5×5:
[[0 0 0 1 0]
 [0 1 0 0 0]
 [0 1 1 0 0]
 [0 0 1 0 1]
 [1 0 0 0 2]]
cel = (4,4), Q.shape = (5,5,4)
```

| Wariant | Sukces | Śr. kroki (udane) |
| --- | --- | --- |
| Agent losowy | **20%** (4/20) | ~16,8 |
| Agent wyuczony (ε_tr=0,2, ε_te=0) | **100%** (20/20) | **~4,8** |

Przykładowa ścieżka z `(0,0)` (8 kroków):  
`(0,0)→(0,1)→(0,2)→(1,2)→(1,3)→(2,3)→(3,3)→(4,3)→(4,4)`

Średnia nagroda z ostatnich 50 epizodów treningu: ~9,1 (przy ε=0,2 w treningu).

### 8.2. Plansza 7×7 (obecny `BOARD` we wzorcu)

```text
cel = (6,6), wolne starty ≈ 35, Q.shape = (7,7,4)
```

| Wariant | Sukces | Śr. kroki (udane) |
| --- | --- | --- |
| Agent losowy | **20%** (4/20) | ~27,8 |
| Agent wyuczony (ε_tr=0,2) | **100%** (20/20) | **~7,05** |

Przykładowa ścieżka z `(0,0)` (**14 kroków**) — idzie lewą/dolną stroną:  
`(0,0)→(1,0)→(2,0)→(2,1)→(3,1)→(4,1)→(5,1)→(6,1)→(6,2)→(6,3)→(5,3)→(5,4)→(5,5)→(6,5)→(6,6)`

| ε w treningu | Sukces w teście | Śr. kroki | Śr. nagroda (ostatnie 50 ep.) |
| --- | --- | --- | --- |
| 0,05 | 100% | ~7,05 | ~9,00 (mniej szumu w treningu) |
| 0,20 | 100% | ~7,05 | ~8,61 |
| 0,30 | 100% | ~7,05 | ~7,64 (więcej szumu przez eksplorację) |

**Wniosek z ε:** na tej mapie wszystkie trzy dają 100% w teście (ε_te=0), ale **wykres treningu** przy większym ε jest „brudniejszy”, a średnia nagroda w epizodach treningowych niższa — bo w treningu agent nadal losuje.

### 8.3. 7×7 + ściana w `(4, 2)` — wynik badania

| | Bez ściany (4,2) | Ze ścianą `BOARD[4,2]=1` |
| --- | --- | --- |
| Sukces wyuczonego | 100% | 100% |
| Śr. kroki (test) | ~7,05 | ~7,25 |
| Kroki ścieżki z (0,0) | **14** | **14** (ta sama długość) |
| Kształt ścieżki | lewa/dół | **inna**: góra/prawo |

Ścieżka ze ścianą (przykład):  
`(0,0)→(0,1)→(0,2)→(0,3)→(1,3)→(1,4)→(1,5)→(2,5)→(2,6)→(3,6)→(4,6)→(4,5)→(5,5)→(6,5)→(6,6)`

**Wniosek kluczowy dla dema/dyskuśji:**  
ściana **nie wydłużyła** tej konkretnej trasy start→cel, a agent i tak nauczył się **innej** drogi. Metryki zbiorcze prawie bez zmian, wizualizacja mocno inna. To pokazuje, że Q-learning nie jest „wyznaczaniem jednej kanonicznej ścieżki”, tylko uczeniem polityki na podstawie doświadczenia.

---

## 9. Wykresy i rysunki — do czego się odnoszą

### `demo_agent_losowy.png`
Krótka wędrówka **bez uczenia**. Często błądzi, odbija się od ścian, może nie dojść do celu.  
**Po co:** punkt odniesienia „brak strategii”.

### `wykres_nagrody*.png` / `wykres_nagrody_referencyjny.png`
- **Szary:** suma nagród w epizodzie (dużo szumu).  
- **Czerwony:** średnia krocząca z 20 epizodów (trend uczenia).  

**Jak czytać sukces:**
1. początek nisko / chaotycznie,  
2. czerwona linia rośnie,  
3. potem plateau wysoko.  

**Dlaczego szare spadki zostają nawet „po nauczeniu”?**  
Bo w **treningu** ε>0 — agent celowo czasem losuje.

Obserwacje z referencji 7×7:
- **ε=0,2** (`wykres_nagrody_referencyjny.png`): zbieżność średniej mniej więcej w okolicy ~100–150 ep.; plateau ~8,5–9.  
- **ε=0,05**: szybciej / czyściej dochodzi blisko 10, mało głębokich spadków pod koniec.  
- **ε=0,30**: plateau niższe (~7–8,5), więcej szumu do końca — silniejsza eksploracja w treningu.

### `sciezka_*.png` / `sciezka_referencyjna.png`
Plansza + trasa:
- pomarańczowy kwadrat = start,  
- niebieska linia = ścieżka,  
- czerwona gwiazda = koniec,  
- zielone pole = cel.  

**Po co:** pokazać wyuczoną politykę „na żywo”, nie tylko procenty.

### `q_table*.npy`
Zapisana tabela Q po treningu (do oddania / odtworzenia).  
Kształt: `(wiersze, kolumny, 4)`.

### Nazewnictwo plików
| Źródło | Przykłady |
| --- | --- |
| Studenci (szkielet) | `q_table.npy`, `wykres_nagrody.png`, `sciezka_wyuczona.png` |
| Wzorzec | `q_table_referencyjna.npy`, `wykres_nagrody_referencyjny.png`, `sciezka_referencyjna.png`, `*_eps005_ref.png`, `*_eps030_ref.png` |

---

## 10. Sprawozdanie — czego oczekujemy (i jak jest zbudowany szablon)

Szablon **nie podaje gotowych liczb** (nagród, α, rozmiaru planszy itd.).  
Studenci mają **odczytać je z kodu / konsoli** — to celowe.

Struktura:
1. Autorzy / uruchomienie  
2. Cel + opis TODO 1–4 własnymi słowami  
3. Eksperyment — tabele parametrów uzupełnione z kodu  
4. Wyniki — wykres, porównanie, ścieżka, opcjonalnie ε  
5. Wnioski (gorsze ruchy w treningu; dlaczego nie testować z ε>0; ograniczenie jednej mapy) + wkład osób  
6. Opcja: większa mapa + ściana (4,2)

### Oddanie (przed demami, Teams)
Kod (`.py`/`.ipynb`) + artefakty + wypełniony szablon.  
Oddawać też rozwiązanie częściowe ze stanem i trudnościami.

---

## 11. Technologia i uruchomienie

- Python 3.10–3.12, CPU  
- `numpy`, `matplotlib`; opcjonalnie Jupyter  
- venv (nie systemowy pip na Debian/Ubuntu)

```bash
python3 -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python warsztat2_szkielet.py                 # zespoły
python warsztat2_rozwiazanie_wzorcowe.py     # my / weryfikacja
```

**Uwaga WSL/headless:** wzorzec używa `matplotlib.use("Agg")` i `plt.close` (bez okna).  
Szkielet używa `plt.show()` — na zajęciach z GUI OK; w czystym WSL może „wisieć” na show.

---

## 12. Pułapki — na co naprowadzać przy stolikach

1. Remisy Q przy zerach → losować, nie ślepy `argmax`.  
2. Przy celu przyszłość = 0.  
3. Test: ε=0, bez update Q.  
4. Niepowodzenia liczą się do %.  
5. Limit 50 = porażka.  
6. Jedna mapa ≠ uniwersalny solver.  
7. Sprawozdanie na bieżąco, bez wklejania całego kodu.  
8. Po zmianie `BOARD` uruchomić ponownie od definicji środowiska.  
9. Przy 7×7 te same 300 ep. mogą ledwo wystarczyć — wolno sugerować więcej epizodów dopiero po obserwacji wykresu.

### Pytania naprowadzające (ściąga)
- „Jaką nagrodę i pozycję daje uderzenie w ścianę?” → −1, bez ruchu.  
- „Co zrobi `argmax`, gdy wszystkie Q=0?” → zawsze ta sama akcja.  
- „Dlaczego na wykresie spadki mimo uczenia?” → ε w treningu.  
- „Dlaczego test z ε=0,2 zafałszuje ocenę?” → miesza strategię z losem.

---

## 13. Wnioski z całego badania (dla nas — „co z tego wynika”)

1. **Metoda działa na małej dyskretnej mapie:** po ~setkach epizodów agent bije losowego o rząd wielkości (np. 20% → 100%).  
2. **Wykres nagród** jest głównym dowodem uczenia; plateau + szum przy ε>0 to normalny obraz treningu.  
3. **ε wpływa głównie na przebieg treningu** (szum / średnia nagroda), a przy dobrej zbieżności test z ε=0 może dać podobną skuteczność dla 0,05 / 0,2 / 0,3.  
4. **Większa mapa** = większe Q, dłuższe ścieżki, trudniejszy start dla agenta losowego — te same hiperparametry nie są magiczne.  
5. **Ściana (4,2)** pokazuje najciekawszy dydaktycznie fakt: **ta sama długość ≠ ta sama ścieżka**; polityka zależy od doświadczenia, nie od „jednej prawdy geometrycznej”.  
6. **Ograniczenie metody:** zmiana mapy → ucz od nowa; tablica Q nie generalizuje na inne labirynty.  
7. **Warsztat jest kompletny operacyjnie:** szkielet ≈ notatnik, wzorzec z komentarzami TODO, sprawozdanie zmusza do czytania kodu, dodatki (ε, 7×7, ściana) są opcjonalne i nie psują zakresu podstawowego.

---

## 14. Co mówić na 10-min prezentacji (skrót)

1. Cel: nauczyć agenta wyjścia z labiryntu Q-learningiem + porównać z losowym.  
2. Stan = pozycja, 4 ruchy, nagrody +10 / −0,1 / −1, limit 50.  
3. Uzupełniacie TODO 1–4, nie budujecie gry.  
4. Harmonogram 90 min.  
5. Oddanie przed demami; sprawozdanie z kodu, nie z pamięci slajdów.  
6. Najpierw podstawa; potem opcjonalnie ε / większa mapa.

---

## 15. Checklist przed zajęciami (Etap I)

- [ ] venv + `pip install -r requirements.txt`  
- [ ] `python warsztat2_rozwiazanie_wzorcowe.py` kończy się w sekundach, powstają PNG/NPY  
- [ ] Próba czasowa: uzupełnienie szkieletu realnie w ~60–75 min  
- [ ] Akceptacja zakresu z prowadzącym  
- [ ] Jeden przewodnik na zespół  
- [ ] Materiały na Teams przed startem 90 min  

---

## 16. Najkrótsza ściąga liderska

```text
Studenci:  szkielet 5×5 → TODO 1–4 → trening → test → wykres + ścieżka → sprawozdanie
My:        wzorzec (teraz 7×7) → weryfikacja / demo / odpowiedzi na pytania
Dodatki:   ε ∈ {0.05, 0.3} ; BOARD 7×7 ; ściana (4,2) → inna ścieżka, podobna długość
Wniosek:   nagroda + eksploracja + test bez uczenia + Q tylko dla tej mapy
```

---

*Notatka zaktualizowana na podstawie aktualnego szkieletu, notatnika, wzorca, szablonu sprawozdania, planów oraz pomiarów referencyjnych (SEED=42).*
