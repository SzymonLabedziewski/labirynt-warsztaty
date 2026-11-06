# Warsztat 2: Agent uczący się przejścia labiryntu (Q-learning)

Pakiet na zajęcia z przedmiotu **Sztuczna inteligencja w grach i rozszerzonej rzeczywistości** (WFiIS AGH).

Ten plik to **instrukcja dla zespołów wykonawczych** — co jest problemem, które pliki są Wasze, jak uruchomić kod, co uzupełnić, co oddać.

---

## 1. Problem — o co chodzi

Macie **agenta** w labiryncie (plansza z wolnymi polami, ścianami i jednym celem).  
Agent widzi tylko swoją pozycję i wybiera ruch: **góra / dół / lewo / prawo**.

**Zadanie:** metodą **tablicowego Q-learningu** nauczyć agenta dochodzenia do wyjścia, a potem **porównać** go z agentem, który rusza całkowicie losowo.

**Czego macie się nauczyć (nie tylko „żeby działało”):**
- jak **nagroda** kieruje zachowaniem,
- po co jest **eksploracja** ($\varepsilon$) podczas treningu,
- dlaczego **test** robi się osobno (bez dalszego uczenia, z $\varepsilon=0$),
- że wynik dotyczy **tej konkretnej planszy**, a nie dowolnego labiryntu.

Silnik gry (plansza, ruch, nagrody, rysowanie) jest **gotowy**. Uzupełniacie oznaczone bloki `# TODO` 1–4.

---

## 2. Co jest w katalogu

### Wasz pakiet startowy (z tego korzystacie)

| Plik | Co to jest |
| --- | --- |
| **`warsztat2_szkielet.py`** | Szkielet zadania (skrypt Python) — TODO 1–4, demo, eksperyment |
| **`warsztat2_szkielet.ipynb`** | **Ten sam kod i te same komentarze** w Jupyter Notebook — wybierzcie `.py` **albo** `.ipynb` |
| **`requirements.txt`** | Biblioteki: `numpy`, `matplotlib` (CPU, bez GPU) |
| **`requirements-notebook.txt`** | To samo + Jupyter (tylko jeśli pracujecie w notatniku) |
| **`Szablon_sprawozdania.md`** | Sprawozdanie — wypełniacie **w trakcie** 90 min; liczby odczytujcie z **własnego kodu / konsoli** |
| **`README.md`** | Ta instrukcja |

### Materiały dodatkowe / prowadzących (nie są Waszym „rozwiązaniem do oddania”)

| Plik | Uwaga |
| --- | --- |
| `Instrukcja_warsztatow.pdf` | Oryginalna instrukcja warsztatów laboratoryjnych |
| `warsztat2_rozwiazanie_wzorcowe.py` | Rozwiązanie wzorcowe zespołu liderskiego — **nie kopiować jako oddanie** |
| `plan.md`, `plan_etap1.md`, `plan_etap2.md` | Konspekty liderskie |
| Pliki `*_referencyjna*`, `*_ref*` (PNG/NPY) | Przykładowe artefakty wzorca — Wasze pliki wynikowe nazywacie inaczej (patrz niżej) |

---

## 3. Instalacja (CPU)

Zalecany Python: **3.10–3.12**.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows (PowerShell):
#   .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Jeśli `python3 -m venv` nie działa: `sudo apt install python3-venv python3-full`.

Opcjonalnie Jupyter:

```bash
pip install -r requirements-notebook.txt
```

---

## 4. Uruchomienie

### Wariant A — skrypt

```bash
python warsztat2_szkielet.py
# albo wprost z venv:
#   .venv/bin/python warsztat2_szkielet.py
#   Windows: .venv\Scripts\python warsztat2_szkielet.py
```

### Wariant B — notatnik

```bash
jupyter notebook warsztat2_szkielet.ipynb
```

Uruchamiajcie komórki **po kolei** (najpierw importy i środowisko, potem TODO, na końcu eksperyment).

### Co zobaczycie przed uzupełnieniem TODO
1. Demo **agenta losowego** (krótka ścieżka, plik `demo_agent_losowy.png`).
2. Komunikat, że TODO 1–4 nie są jeszcze zaimplementowane.

Po uzupełnieniu TODO i ponownym uruchomieniu powinny powstać m.in. wyniki porównania w konsoli oraz pliki wynikowe (sekcja 7).

---

## 5. Zasady środowiska (krótko)

Oznaczenia w `BOARD`:
- `0` — pole wolne  
- `1` — ściana  
- `2` — cel (wyjście)

**Nagrody** (funkcja `step` w kodzie — sprawdźcie dokładne wartości u siebie):
- wejście na cel → duża nagroda dodatnia  
- zwykły dozwolony ruch → mała kara (koszt czasu)  
- ściana / poza planszę → kara; agent **zostaje** na miejscu  

Epizod kończy się po osiągnięciu celu albo po limicie kroków (`MAX_STEPS`).

Domyślna plansza w szkielecie to **5×5**. Listy startów (`VALID_STARTS`, `TEST_STARTS`) i kształt tablicy Q zależą od `BOARD` — po zmianie planszy uruchomcie kod od nowa od definicji środowiska.

---

## 6. Zakres pracy (ok. 90 min)

### Zakres podstawowy (priorytet)

| Krok | Gdzie w kodzie | Co zrobić |
| --- | --- | --- |
| **TODO 1** | `create_q_table` | Tablica Q wypełniona zerami; kształt `(*BOARD.shape, 4)` |
| **TODO 2** | `choose_action` | Wybór $\varepsilon$-zachłanny; **remisy Q rozstrzygajcie losowo** |
| **TODO 3** | wewnątrz `train_q_learning` | Aktualizacja Q; przy celu wartość przyszła = 0 |
| **TODO 4** | `evaluate_agent` | Test na `TEST_STARTS`: $\varepsilon=0$, **bez** aktualizacji Q |
| Eksperyment | `run_basic_experiment` | Trening → porównanie z agentem losowym → wykres → ścieżka |
| Sprawozdanie | `Szablon_sprawozdania.md` | Wypełniać na bieżąco; parametry **odczytać z kodu**, nie zgadywać |

Parametry bazowe treningu (sprawdźcie w wywołaniu `train_q_learning` / komentarzach):  
ok. **300** epizodów, $\alpha$, $\gamma$, $\varepsilon$ treningowe, limit kroków, lista **20** startów testowych, `SEED`.

### Zakres dodatkowy (opcjonalnie — gdy podstawa działa)

1. **Porównanie $\varepsilon$:** odkomentujcie `run_extension_experiment()` (trening m.in. dla $0{,}05$ i $0{,}3$).  
2. **Większa plansza (np. 7×7):** przykład w komentarzu przy `BOARD`; pytania i checklista w **sekcji 6** szablonu sprawozdania.  
3. **Podpowiedź badawcza:** na 7×7 ściana w polu `(4, 2)` (`BOARD[4, 2] = 1`) — patrz sekcja **6.4** sprawozdania (inna ścieżka przy podobnej długości trasy).

**Najpierw** dokończcie zakres podstawowy. Dodatki nie zastępują działającego TODO i oddania.

---

## 7. Nazwy plików wynikowych (Wasze)

Po udanym zakresie podstawowym typowo powstają:

| Plik | Znaczenie |
| --- | --- |
| `q_table.npy` | Wyuczona tablica Q |
| `wykres_nagrody.png` | Przebieg sumy nagród w epizodach (+ średnia krocząca) |
| `sciezka_wyuczona.png` | Przykładowa ścieżka wyuczonego agenta |
| `demo_agent_losowy.png` | Demo agenta losowego (na start) |

Przy zakresie dodatkowym użyjcie **osobnych nazw**, żeby nie nadpisać wyników bazowych, np.:
- `wykres_nagrody_eps005.png`, `wykres_nagrody_eps030.png`
- `wykres_nagrody_7x7.png`, `sciezka_wyuczona_7x7.png`, `q_table_7x7.npy`

### Jak czytać wykres nagród
- linia szara — nagroda w pojedynczym epizodzie (dużo szumu),  
- linia czerwona — średnia z kilku epizodów (widać, czy agent się uczy).  
Spadki mimo uczenia bywają normalne, gdy w **treningu** $\varepsilon > 0$ (eksploracja).

---

## 8. Sprawozdanie

Plik: **`Szablon_sprawozdania.md`**.

- Piszecie **w trakcie** zajęć (nie „potem z pamięci”).  
- **Nie** wklejajcie całego kodu — opiszcie zmiany w TODO i wyniki.  
- Liczby (nagrody, $\alpha$, $\gamma$, $\varepsilon$, rozmiar planszy, limity…) **odczytajcie z kodu / konsoli / zapisanych plików**.  
- We wnioskach wyjaśnijcie m.in.: gorsze ruchy przy treningu, dlaczego test z $\varepsilon>0$ zafałszowałby ocenę, ograniczenie do jednej planszy.

---

## 9. Oddanie (przed demami, MS Teams)

1. Uzupełniony kod: `warsztat2_szkielet.py` **lub** `warsztat2_szkielet.ipynb`  
2. Artefakty: co najmniej `q_table.npy` + wykres nagród + ścieżka (oraz inne, jeśli robiliście dodatki)  
3. Wypełniony `Szablon_sprawozdania.md`  

Oddajcie także **rozwiązanie częściowe**, jeśli nie skończyliście — ze wskazaniem stanu prac i napotkanych trudności.

Na demo (ok. 5–6 min): pokażcie przejście agenta, liczby losowy vs wyuczony i krótki wniosek.

---

## 10. Proponowany rytm 90 minut

| Czas | Co robić |
| --- | --- |
| 0–10 min | Instalacja / start, demo losowe, otwarcie sprawozdania, podział ról |
| 10–25 min | TODO 1–2 |
| 25–55 min | TODO 3 + trening + wykres |
| 55–75 min | TODO 4 + porównanie (+ ewentualnie $\varepsilon$) |
| 75–90 min | Ścieżka, wnioski, wrzucenie plików na Teams |

Przy stoliku jest **przewodnik** z zespołu liderskiego: pomaga i naprowadza, ale **nie wykonuje zadania za zespół**.

---

## 11. Typowe pułapki

- Przy starcie wiele wartości Q = 0 — zwykły `argmax` zawsze wybierze tę samą akcję; **remisy losujcie**.  
- Po wejściu na cel nie bierzcie „przyszłości” z `max Q` pola celu — wartość przyszła = 0.  
- W teście: $\varepsilon = 0$ i **żadnej** aktualizacji Q.  
- Niepowodzenia (limit kroków) też wchodzą do procentu skuteczności.  
- Po zmianie `BOARD` uruchomcie ponownie definicję środowiska (nie tylko sam trening na starych zmiennych).

---

## 12. Szybki start

```text
1. python3 -m venv .venv  →  aktywacja  →  pip install -r requirements.txt
2. python warsztat2_szkielet.py   (lub notatnik)
3. Uzupełnij TODO 1 → 2 → 3 → 4
4. Uruchom ponownie → sprawdź konsolę + PNG/NPY
5. Wypełnij Szablon_sprawozdania.md (dane z kodu)
6. Oddaj na Teams przed demami
```

Powodzenia.
