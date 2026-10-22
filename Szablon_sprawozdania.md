# Sprawozdanie — Warsztat 2: Agent uczący się przejścia labiryntu

**Przedmiot:** Sztuczna inteligencja w grach i rozszerzonej rzeczywistości  

> Wypełniajcie **w trakcie** zajęć. Dane bierzcie z własnego kodu / uruchomienia.  
> Nie wklejajcie całego kodu — krótki opis zmian i wyniki.

---

## 1. Temat i autorzy

| Pole | Wpis |
| --- | --- |
| Numer zespołu | |
| Uczestnicy | |
| Data | |
| Przewodnik (zespół liderski) | |
| Wersja kodu (plik / commit / archiwum) | |
| Jak uruchomić rozwiązanie | |

---

## 2. Cel i wykonane zmiany

**Cel ćwiczenia** (własnymi słowami):

**Uzupełnione TODO** — co zrobiliście (bez paste’owania kodu):

| TODO | Funkcja | Co uzupełniliście / jak działa |
| --- | --- | --- |
| 1 | `create_q_table` | |
| 2 | `choose_action` | |
| 3 | aktualizacja w `train_q_learning` | |
| 4 | `evaluate_agent` | |

*Pytania naprowadzające (opcjonalnie, jeśli utkniecie przy opisie):*  
jaki kształt ma Q? kiedy losujecie akcję, a kiedy bierzecie z Q? co dzieje się przy remisie Q? czym test różni się od treningu?

---

## 3. Eksperyment

### 3.1. Środowisko
| Element | Wartość / opis | Źródło (gdzie w kodzie) |
| --- | --- | --- |
| Plansza (rozmiar) | | |
| Oznaczenia pól | | |
| Cel (współrzędne) | | |
| Limit kroków | | |
| Akcje | | |

### 3.2. Nagrody
| Sytuacja | Nagroda | Skutek dla pozycji |
| --- | --- | --- |
| Cel | | |
| Zwykły ruch | | |
| Ściana / poza planszę | | |

### 3.3. Parametry treningu (zakres podstawowy)
| Parametr | Wartość | Źródło |
| --- | --- | --- |
| Epizody | | |
| $\alpha$ | | |
| $\gamma$ | | |
| $\varepsilon$ (trening) | | |
| `SEED` | | |

### 3.4. Test
* Lista startów (nazwa, długość, czy ustalona):  
* Porównywane warianty:  
* Zakres dodatkowy (jeśli był) — co zmienialiście:

---

## 4. Wyniki

### 4.1. Przebieg uczenia
* Plik wykresu:  
* Krótka obserwacja (początek / środek / koniec):

### 4.2. Porównanie na liście testowej

| Wariant | Odsetek dotarć (%) | Udane / wszystkie | Śr. kroków (udane) | Niepowodzenia |
| --- | :---: | :---: | :---: | :---: |
| Agent losowy | | | | |
| Agent wyuczony | | | | |

Ustawienia wyuczonego w teście ($\varepsilon$, czy aktualizacja Q):

Uwagi / trudności:

### 4.3. Przykładowa ścieżka
* Plik / współrzędne:  
* Liczba kroków:  
* Czy osiągnięto cel:

### 4.4. Zakres dodatkowy — $\varepsilon$ (jeśli robiony)

| $\varepsilon$ w treningu | Odsetek w teście (%) | Śr. kroków | Obserwacje |
| --- | :---: | :---: | --- |
| | | | |
| | | | |

---

## 5. Wnioski

Oczekujemy krótkich odpowiedzi na podstawie **Waszych** wyników:

1. Co zadziałało / co nie?  
2. Dlaczego w treningu agent może wykonywać gorsze ruchy?  
3. Dlaczego test z $\varepsilon > 0$ nie mierzyłby samej wyuczonej strategii?  
4. Jakie ograniczenia ma zastosowana metoda?  
5. Wkład członków zespołu:
   *  
   *  
   *  
   *  

---

## 6. Zakres dodatkowy — inna / większa plansza (opcjonalnie)

Wypełnijcie **tylko jeśli** zrobiliście ten wariant po działającym zakresie podstawowym.

| Pytanie | Odpowiedź |
| --- | --- |
| Jaka plansza (rozmiar / krótki opis)? | |
| Co zmieniliście względem wersji bazowej? | |
| Skuteczność i śr. kroki vs plansza bazowa? | |
| Czy trzeba było zmienić parametry treningu? Jakie i z jakim efektem? | |
| Co widać na wykresie / ścieżce względem bazy? | |
| Główny wniosek z tego porównania? | |

*Jeśli badaliście wpływ dodatkowej ściany na ścieżkę przy podobnej liczbie ruchów:*  
czy ścieżka się zmieniła? czy metryki (sukces, śr. kroki) też? jak to interpretujecie?
