# CZĘŚĆ 1: Scenariusz prezentacji (pod slajdy 1–8)

### ⏱️ | Slajd 1–2: Temat i pętla Uczenia ze Wzmocnieniem (RL)

* **Co powiedzieć (cel):**
  > „Dzisiaj nie programujemy gry od zera – silnik labiryntu i rysowanie macie już gotowe. Waszym zadaniem jest zaprogramowanie **mózgu agenta** metodą **tablicowego Q-learningu**, żeby sam nauczył się wychodzić z labiryntu, i porównanie go z agentem losowym.”
* **Jak wytłumaczyć ideę RL (Slajd 2):**
  * Użyj **analogii szczura w labiryncie / gry w ciemno na punkty**: Agent nie ma mapy w głowie i nikt nie mówi mu „idź w prawo”. Zna tylko swoją aktualną pozycję (`state`), wybiera kierunek ruchu (`action`), a środowisko (`step`) odpowiada mu: *„jesteś teraz na polu X i dostajesz tyle a tyle punktów (*`reward`*)”*.
  * Podkreśl, że to **tablicowy Q-learning** – cała wiedza agenta to zwykła tabela liczb w `numpy` (bez sieci neuronowych i bez GPU).

---

### ⏱️ | Slajd 3 i 5: Środowisko i System Nagród (Zasady gry)

* **Jak wyjaśnić planszę (**`BOARD` **– Slajd 3):**
  * Domyślna plansza to siatka **5×5** (w kodzie macierz `BOARD`):
    * `0` – wolne pole (po nich chodzimy i z nich losujemy starty),
    * `1` – ściana (przeszkoda),
    * `2` – cel / wyjście (w rogu `(4, 4)`).
  * **Stan (**`state`**)**: krotka `(wiersz, kolumna)`, np. `(0, 0)`.
  * **Akcje (**`action`**)**: 4 liczby: `0` (Góra), `1` (Dół), `2` (Lewo), `3` (Prawo).
  * **Koniec epizodu**: wejście na cel **LUB** przekroczenie limitu `MAX_STEPS = 50` kroków (wtedy uznajemy epizod za porażkę!).
* **Jak wytłumaczyć System Nagród (Slajd 5 – kluczowe pytanie do sprawozdania!):**
  * **Wejście na cel (**`+10.0`**)**: wielka nagroda, koniec epizodu (`reached_goal = True`).
  * **Uderzenie w ścianę lub wyjście poza mapę (**`-1.0`**)**: spora kara i **ważne: agent zostaje na tym samym polu!** (`next_state = state`).
  * **Zwykły dozwolony ruch (**`-0.1`**)**: mała kara za każdy krok.
  * *Jak to zilustrować grupie:* „Po co dajemy `-0.1` za zwykły ruch, skoro sam cel daje `+10`? Gdyby ruch kosztował `0`, agent mógłby kręcić się w kółko przez 40 kroków i na końcu wejść do celu z taką samą nagrodą `+10`. Kara `-0.1` to **koszt czasu** – wymusza szukanie **najkrótszej** ścieżki”.

---

### ⏱️ | Slajd 4: Tablica Q, wybór ruchu i wzór aktualizacji (Serce algorytmu)

To najważniejszy moment prezentacji. Wyjaśnij 3 rzeczy:

1. **Czym jest Tablica** `Q` **(TODO 1):**
   * *Ilustracja:* „Wyobraźcie sobie, że na każdym kafelku podłogi w labiryncie rysujecie kredą 4 strzałki (Góra, Dół, Lewo, Prawo) i przy każdej strzałce piszecie ocenę – jak bardzo opłaca się tam pójść”.
   * W kodzie to tablica 3D o wymiarach `(wiersze, kolumny, 4)`, czyli dla planszy 5×5 ma kształt `(5, 5, 4)`. Na starcie wypełniamy ją samymi **zerami** (agent nic nie wie).
2. **Jak agent wybiera ruch – strategia $\varepsilon$-zachłanna / epsilon-greedy (TODO 2):**
   * Rzucamy nieuczciwą monetą (losujemy liczbę z przedziału $[0, 1)$):
     * Z prawdopodobieństwem $\varepsilon$ (w treningu `0.2`, czyli 20% czasu) – **Eksploracja**: agent ignoruje tablicę `Q` i wybiera losowy kierunek (żeby odkrywać nowe ścieżki).
     * W pozostałych 80% przypadków ($1 - \varepsilon$) – **Eksploatacja**: agent wybiera kierunek z najwyższą wartością w tablicy `Q` dla obecnego pola.
   * ⚠️ **Pierwsza pułapka do głośnego ostrzeżenia:** „Na początku wszystkie wartości w `Q` wynoszą `0`. Jeśli użyjecie zwykłego `np.argmax()`, Python przy remisie **zawsze zwróci indeks** `0` **(Góra)** i agent będzie ciągle bił głową w górną ścianę! Dlatego przy remisie najlepszych akcji musicie **wylosować jedną z nich**”.
3. **Jak agent się uczy – aktualizacja Q-learning (TODO 3):**
   * Po wykonaniu ruchu z pola $s$ akcją $a$ na pole $s'$, korygujemy ocenę strzałki:
     $$
     \text{target} = r + \gamma \cdot \max_{a'} Q(s', a')
     $$
     $$
     Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \big(\text{target} - Q(s, a)\big)
     $$
   * ⚠️ **Druga pułapka do głośnego ostrzeżenia:** „Jeśli ruch doprowadził do **celu** (`reached_goal == True`), epizod się kończy – nie ma już kolejnych ruchów! Wtedy przyszłość wynosi `0`, czyli $\text{target} = r$ (samo `+10.0`), bez części z $\gamma$”.

---

### ⏱️ | Slajd 6: Eksploracja vs Ewaluacja (Trening ≠ Test)

* **Co trzeba mocno rozgraniczyć (TODO 4 i wnioski w sprawozdaniu):**
  * **W treningu (**`train_q_learning`**)**: $\varepsilon = 0.2$ (agent czasem specjalnie błądzi, żeby poznawać mapę) i w każdym kroku aktualizujemy tablicę `Q`.
  * **W teście / ewaluacji (**`evaluate_agent`**)**: Sprawdzamy, czego agent się nauczył na 20 ustalonych pozycjach (`TEST_STARTS`). Tutaj:
    1. Wyłączamy losowość: ustawiamy **$\varepsilon = 0.0$** (gdyby $\varepsilon > 0$, ocenialibyśmy mieszankę wiedzy i ślepego losu!).
    2. **Nie aktualizujemy tablicy** `Q` (test to egzamin, na egzaminie zamrażamy wiedzę).
    3. Do procentu skuteczności (`success_rate`) wliczamy wszystkie 20 prób (porażki po 50 krokach też!), a średnią liczbę kroków (`avg_steps_success`) liczymy tylko dla udanych przejść.

---

### ⏱️ | Slajd 7 i 8: Harmonogram 90 min, pliki i sprawozdanie

* **Pokaż, z czego korzystają:**
  * Wybierają **jeden** plik do pracy: `warsztat2_szkielet.py` **albo** `warsztat2_szkielet.ipynb` (mają identyczny kod i podpowiedzi).
  * Od razu na starcie otwierają `Szablon_sprawozdania.md` (lub `.docx`) i **wypełniają go równolegle z kodowaniem**.
* **Zasada sprawozdania:**
  * „W szablonie sprawozdania nie ma wpisanych parametrów – musicie je **odczytać wprost z kodu** (z funkcji `step`, `train_q_learning`, stałych `MAX_STEPS`, `SEED`). Nie wklejacie tam surowego kodu, tylko opisujecie własnymi słowami co robią TODO 1–4 i wpisujecie wyniki z konsoli”.
* **Podział na zakres podstawowy i dodatkowy:**
  * Najpierw obowiązkowa podstawa (TODO 1–4 na planszy 5×5, wykres nagród, ścieżka, tabela porównania losowy vs wyuczony).
  * Dopiero gdy to działa i pliki są zapisane: zakres dodatkowy (`run_extension_experiment()` dla $\varepsilon = 0.05$ i $0.30$ oraz zmiana `BOARD` na 7×7 / inne mapy z osobnym zapisem plików, żeby nie nadpisać podstawy!).
* **Co oddają na MS Teams przed demami (75–90 min):**
  * Kod (`.py` lub `.ipynb`), wygenerowane pliki (`q_table.npy`, `wykres_nagrody.png`, `sciezka_wyuczona.png`) oraz uzupełnione sprawozdanie.

---

# CZĘŚĆ 2: Ściąga zależności – „Co za co odpowiada i co na co wpływa”

Ta tabela przyda Ci się podczas prezentacji oraz gdy studenci będą pytać przy stolikach:

### 1. Hiperparametry algorytmu i środowiska

| Zmienna w kodzie                    | Co oznacza                                     | Za co odpowiada                                                                          | Co się stanie, gdy ją zmienimy (wpływ na wynik i wykres)                                                                                                                                                                                                                                                                                                                        |
| :---------------------------------- | :--------------------------------------------- | :--------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `alpha` **($\alpha = 0.1$)**        | Tempo uczenia (*learning rate*)                | Jak mocno nowe doświadczenie nadpisuje starą ocenę w tablicy `Q`.                        | **Za małe (np. 0.01):** agent uczy się bardzo wolno, 300 epizodów nie wystarczy. **Za duże (np. 0.9):** nadpisuje wiedzę pojedynczymi skokami, niestabilność.                                                                                                                                                                                                                   |
| `gamma` **($\gamma = 0.95$)**       | Współczynnik dyskontowania (*discount factor*) | Jak bardzo agent dba o **przyszłe** nagrody względem nagrody natychmiastowej.            | **Bliskie 0:** agent jest krótkowzroczny (widzi tylko karę `-0.1` przed nosem, informacja o `+10` z celu nie rozlewa się na wcześniejsze pola). **Bliskie 1 (0.95):** nagroda `+10` z celu propaguje się wstecz przez cały labirynt aż do pola `(0,0)`.                                                                                                                         |
| `epsilon` **($\varepsilon = 0.2$)** | Współczynnik eksploracji                       | Jak często w treningu agent wybiera losowy ruch zamiast najlepszego z `Q`.               | **Małe ($\varepsilon=0.05$):** wykres nagród w treningu jest bardzo gładki i wysoki (~9.0), ale na trudnej mapie agent może utknąć w suboptymalnej trasie. **Duże ($\varepsilon=0.30$):** wykres w treningu ma dużo „szumu” i spadków (średnia ~7.6), bo agent co chwilę celowo skręca w ściany/boczki, ale dobrze poznaje mapę. W teście ($\varepsilon=0$) oba mogą mieć 100%! |
| `episodes` **(**`300`**)**          | Liczba epizodów treningowych                   | Ile pełnych prób przejścia labiryntu wykonuje agent podczas nauki.                       | Na mapie 5×5 średnia nagroda stabilizuje się już ok. 80–120 epizodu. Na większej mapie (7×7) 300 epizodów to absolutne minimum – czasem warto zwiększyć.                                                                                                                                                                                                                        |
| `MAX_STEPS` **(**`50`**)**          | Limit kroków w jednym epizodzie                | Zabezpiecza przed nieskończonym błądzeniem w pętli.                                      | Gdy agent utknie (np. odbija się od ściany), po 50 krokach epizod jest przerywany i liczony jako **porażka** (`reached_goal = False`).                                                                                                                                                                                                                                          |
| `SEED` **(**`42`**)**               | Ziarno generatora losowego                     | Zapewnia powtarzalność losowania startów testowych (`TEST_STARTS`) i przebiegu treningu. | Dzięki niemu porównanie agenta losowego, wyuczonego oraz różnych wartości $\varepsilon$ odbywa się na **dokładnie tych samych 20 pozycjach startowych**.                                                                                                                                                                                                                        |

---

### 2. Zmienne programowe w kodzie (`warsztat2_szkielet.py`)

```47:72:c:\Users\DELL\Documents\SI_war_labirynt\warsztat2_szkielet.py
BOARD = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 0, 2],
])

GOAL_STATE = tuple(zip(*np.where(BOARD == 2)))[0]  # automatycznie z BOARD
MAX_STEPS = 50

# Akcje: 0 = Góra, 1 = Dół, 2 = Lewo, 3 = Prawo
ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
ACTION_NAMES = {0: "Góra", 1: "Dół", 2: "Lewo", 3: "Prawo"}

VALID_STARTS = [
    (r, c)
    for r in range(BOARD.shape[0])
    for c in range(BOARD.shape[1])
    if BOARD[r, c] == 0
]
```

* `BOARD` – macierz 2D definiująca labirynt. **Co od niej zależy:** automatycznie wylicza się z niej `GOAL_STATE`, lista dozwolonych pól `VALID_STARTS`, lista 20 startów testowych `TEST_STARTS` oraz rozmiar tablicy `Q` (`(*BOARD.shape, 4)`). *Wskazówka dla studentów: jeśli w zadaniu dodatkowym zmienią* `BOARD` *na 7×7 w Jupyter Notebooku, muszą ponownie uruchomić komórkę definiującą środowisko, a nie sam trening!*
* `VALID_STARTS` **vs** `TEST_STARTS`:
  * W **treningu** `reset_env()` losuje za każdym razem dowolne wolne pole z `VALID_STARTS` – dzięki temu agent uczy się wychodzić z **każdego miejsca** na planszy, a nie tylko z `(0,0)`.
  * W **ewaluacji** używamy stałej listy 20 pól `TEST_STARTS` (gdzie pierwszym polem `TEST_STARTS[0]` jest zawsze `(0,0)`, z którego rysowana jest potem ścieżka `sciezka_wyuczona.png`).
* `Q[row, col, action]` – 3-wymiarowa tablica `numpy`. Dwa pierwsze indeksy to współrzędne stanu `state = (row, col)`, a trzeci to numer akcji `0..3`. Wycinek `Q[state[0], state[1], :]` zwraca wektor 4 ocen dla czterech kierunków z danego pola.

---

# CZĘŚĆ 3: Jak zilustrować wykresy i ograniczenia metody (kontekst sprawozdania)

Podczas prezentacji (lub podpowiadając w trakcie zajęć) warto wyjaśnić 3 zjawiska, o które wprost pyta **Sekcja 4, 5 i 6** `Szablon_sprawozdania.md`:

1. **Jak czytać** `wykres_nagrody.png` **(Szara linia vs Czerwona linia):**
   * **Czerwona linia** (średnia krocząca z 20 epizodów) pokazuje trend: na początku jest nisko/ujemnie (agent błądzi i bije w ściany), potem szybko rośnie i wypłaszcza się wysoko (ok. `+8.5` do `+9.2`).
   * **Szara linia** (pojedyncze epizody) nawet w 250–300 epizodzie ma pojedyncze ostre spadki w dół. **Dlaczego?** Bo w treningu cały czas mamy $\varepsilon = 0.2$ (20% losowych ruchów, np. przypadkowe wejście w ścianę za `-1.0`) oraz losujemy różne punkty startowe (z dalszego pola trzeba wykonać więcej kroków po `-0.1`).
2. **Kluczowe ograniczenie tablicowego Q-learningu (Pytanie 5.4 w sprawozdaniu):**
   * Tablica `Q` pamięta współrzędne konkretnych pól `(wiersz, kolumna)` **tylko dla tej jednej mapy**.
   * Jeśli przesuniemy ścianę albo zmienimy rozmiar planszy, wyuczona tablica `Q` staje się bezużyteczna – agent nie „rozumie” pojęcia ściany, pamięta tylko „na polu (1,2) idź w prawo”. Dla nowej mapy trzeba przeprowadzić trening od nowa.
3. **Ciekawostka badawcza z większą mapą i ścianą** `BOARD[4, 2] = 1` **(Sekcja 6 sprawozdania):**
   * Jeśli zespół dojdzie do części dodatkowej (plansza 7×7), postawienie jednej dodatkowej ściany w `(4,2)` często **nie zmienia minimalnej liczby kroków do celu ani 100% skuteczności**, ale sprawia, że agent wybiera **zupełnie inną trasę** (np. górą zamiast dołem).
   * *Wniosek:* W labiryncie istnieje wiele równie dobrych ścieżek (optimum). Q-learning nie szuka „jednej oficjalnej drogi geometrycznej”, tylko wzmacnia tę trasę, którą jako pierwszą skutecznie wyeksplorował podczas losowych prób.

---

# Krótka checklista na tablicę / slajd końcowy (co zostawić wyświetlonego na 90 min pracy)

1. **Plik roboczy:** `warsztat2_szkielet.py` lub `warsztat2_szkielet.ipynb` + otwarty `Szablon_sprawozdania.md`.
2. **Kolejność pracy:**
   * `0–10 min`: Uruchomienie dema losowego (`demo_agent_losowy.png`), wpisanie parametrów z kodu do Sekcji 3 sprawozdania.
   * `10–25 min`: **TODO 1** (`create_q_table` – zera `(5,5,4)`) i **TODO 2** (`choose_action` – $\varepsilon$-greedy + **losowanie remisów!**).
   * `25–55 min`: **TODO 3** (`train_q_learning` – dla celu `target = reward`, inaczej z `gamma * max(Q)`).
   * `55–75 min`: **TODO 4** (`evaluate_agent` – $\varepsilon = 0.0$, **bez aktualizacji Q**, liczymy wszystkie 20 prób).
   * `75–90 min`: Wnioski w sprawozdaniu, opcjonalnie zakres dodatkowy ($\varepsilon \in 0.05, 0.30$, mapa 7×7), wrzucenie paczki na MS Teams.