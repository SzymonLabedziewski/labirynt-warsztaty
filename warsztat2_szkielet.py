"""
WARSZTAT 2: AGENT UCZĄCY SIĘ PRZEJŚCIA LABIRYNTU (Q-LEARNING)
Sztuczna inteligencja w grach i rozszerzonej rzeczywistości – WFiIS AGH

SZKIELET DLA ZESPOŁU WYKONAWCZEGO
Uzupełnij bloki oznaczone TODO 1–4, potem uruchom:
    python warsztat2_szkielet.py
albo (ten sam kod / te same komentarze):
    jupyter notebook warsztat2_szkielet.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# =====================================================================
# 1. DEFINICJA ŚRODOWISKA (GOTOWE)
# =====================================================================
# Oznaczenia pól na planszy:
# 0 – pole wolne, 1 – ściana, 2 – cel (wyjście)
#
# ► Możesz zmienić BOARD na dowolną planszę (dowolny rozmiar).
#   VALID_STARTS i TEST_STARTS zostaną wygenerowane automatycznie.
#   Pamiętaj, żeby cel (2) pozostał w planszy i był osiągalny.
#
# Zakres dodatkowy (dopiero gdy TODO 1–4 + wykresy/ścieżka działają):
#   zamień BOARD np. na 7×7, uruchom ponownie trening/test i porównaj wyniki
#   z planszą bazową (patrz sekcja 6 w Szablon_sprawozdania.md).
#   Przykład 7×7 (cel osiągalny) – odkomentuj i zakomentuj BOARD 5×5 poniżej:
# BOARD = np.array([
#     [0, 0, 0, 0, 1, 0, 0],
#     [0, 1, 1, 0, 0, 0, 1],
#     [0, 0, 0, 0, 1, 0, 0],
#     [1, 0, 1, 1, 0, 1, 0],
#     [0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 1],
#     [0, 0, 0, 0, 1, 0, 2],
# ])
# Podpowiedź badawcza (sekcja 6.4 sprawozdania): na 7×7 ustaw dodatkowo
# BOARD[4, 2] = 1  (ściana w wierszu 4, kolumnie 2). Często min. liczba ruchów
# do celu się nie zmienia, a wyuczona ścieżka i tak bywa inna — bo Q-learning
# wzmacnia ćwiczone trasy (wiele optimum, ε, remisy Q), a nie „oficjalną” drogę.
# Porównaj rysunek ścieżki oraz % sukcesu / śr. kroki przed i po tej ścianie.

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


def generate_test_starts(board=None, n=20, seed=SEED):
    """
    Generuje listę n pozycji startowych (z powtórzeniami) spośród wolnych
    pól planszy (board[r, c] == 0).

    Parametry
    ----------
    board : np.ndarray lub None
        Plansza do analizy. Domyślnie używa globalnego BOARD.
    n : int
        Liczba pozycji testowych do wylosowania.
    seed : int
        Ziarno generatora – gwarantuje powtórzalność wyników.

    Zwraca
    -------
    list of (int, int)
        Lista n krotek (row, col), gdzie board[row, col] == 0.
    """
    if board is None:
        board = BOARD
    free_cells = [
        (r, c)
        for r in range(board.shape[0])
        for c in range(board.shape[1])
        if board[r, c] == 0
    ]
    if not free_cells:
        raise ValueError("Brak wolnych pól na planszy – sprawdź definicję BOARD.")
    # Pierwsza pozycja to zawsze lewy górny róg (free_cells[0] = (0,0) jeśli wolne),
    # co gwarantuje spójność z kodem rysującym ścieżkę z pierwszego startu.
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(free_cells), size=n - 1, replace=True)
    return [free_cells[0]] + [free_cells[i] for i in indices]


# Automatycznie generowana lista pozycji testowych na podstawie aktualnego BOARD.
# Po zmianie BOARD wystarczy ponownie uruchomić skrypt / komórkę – TEST_STARTS
# zaktualizuje się automatycznie, zawsze zawierając wyłącznie wolne pola nowej planszy.
TEST_STARTS = generate_test_starts()


def reset_env(start_pos=None):
    """Resetuje środowisko do wybranej lub losowej dozwolonej pozycji startowej."""
    if start_pos is not None:
        if BOARD[start_pos[0], start_pos[1]] != 0:
            raise ValueError(f"Niedozwolona pozycja startowa: {start_pos}")
        return start_pos
    idx = np.random.choice(len(VALID_STARTS))
    return VALID_STARTS[idx]


def step(state, action):
    """
    Wykonuje ruch w środowisku.
    Zwraca: (next_state, reward, reached_goal)

    Nagrody rozłączne:
      +10.0 za wejście na cel
       -1.0 za próbę wejścia w ścianę lub poza planszę (agent zostaje na miejscu)
       -0.1 za zwykły dozwolony ruch
    """
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc

    rows, cols = BOARD.shape
    if nr < 0 or nr >= rows or nc < 0 or nc >= cols or BOARD[nr, nc] == 1:
        return state, -1.0, False

    next_state = (nr, nc)
    if BOARD[nr, nc] == 2:
        return next_state, 10.0, True
    return next_state, -0.1, False


def draw_path(path, title="Ścieżka agenta", filename=None):
    """Rysuje planszę oraz ścieżkę przebytą przez agenta."""
    rows_b, cols_b = BOARD.shape
    fig, ax = plt.subplots(figsize=(cols_b, rows_b))
    display_map = np.zeros((rows_b, cols_b, 3))

    for r in range(rows_b):
        for c in range(cols_b):
            if BOARD[r, c] == 1:
                display_map[r, c] = [0.2, 0.2, 0.2]
            elif BOARD[r, c] == 2:
                display_map[r, c] = [0.2, 0.8, 0.2]
            else:
                display_map[r, c] = [0.95, 0.95, 0.95]

    ax.imshow(display_map)
    ax.set_xticks(np.arange(-0.5, cols_b, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows_b, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.set_xticks(range(cols_b))
    ax.set_yticks(range(rows_b))

    if path:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, marker="o", color="royalblue", linewidth=2, markersize=8, label="Ścieżka")
        ax.plot(cols[0], rows[0], marker="s", color="orange", markersize=12, label="Start")
        ax.plot(cols[-1], rows[-1], marker="*", color="red", markersize=14, label="Koniec")

    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Zapisano: {filename}")
    plt.show()


def demo_random_agent(max_steps=30):
    """Pokazuje krótkie przejście agenta losowego (etap 0–10 min)."""
    state = reset_env((0, 0))
    path = [state]
    total_reward = 0.0
    for _ in range(max_steps):
        action = int(np.random.randint(4))
        state, reward, done = step(state, action)
        total_reward += reward
        path.append(state)
        if done:
            break
    print(
        f"Agent losowy ze startu (0,0): kroki={len(path) - 1}, "
        f"nagroda={total_reward:.1f}, cel={path[-1] == GOAL_STATE}"
    )
    draw_path(path, title="Przykład: agent losowy", filename="demo_agent_losowy.png")


# =====================================================================
# 2. MIEJSCA DO UZUPEŁNIENIA (CZĘŚĆ ROBOCZA: 90 MIN)
# =====================================================================

# --- KROK 1 (10–25 min): tablica Q i wybór epsilon-zachłanny ---

def create_q_table():
    """
    TODO 1: Utwórz i zwróć tablicę Q wypełnioną zerami
    dla wszystkich par stan–akcja (plansza o rozmiarze BOARD, 4 akcje).
    Oczekiwany kształt: (*BOARD.shape, 4).
    """
    # UZUPEŁNIJ TUTAJ:
    raise NotImplementedError("TODO 1: create_q_table")


def choose_action(Q, state, epsilon):
    """
    TODO 2: Wybór akcji metodą epsilon-zachłanną.
    - Z prawdopodobieństwem epsilon wylosuj akcję spośród {0, 1, 2, 3}.
    - W przeciwnym razie wybierz akcję o największej wartości Q dla stanu.
    - Remisy rozstrzygaj LOSOWO (np. np.flatnonzero / np.where + np.random.choice).
    """
    # UZUPEŁNIJ TUTAJ:
    raise NotImplementedError("TODO 2: choose_action")


# --- KROK 2 (25–55 min): aktualizacja Q-learning ---

def train_q_learning(episodes=300, alpha=0.1, gamma=0.95, epsilon=0.2):
    """
    TODO 3: Uzupełnij aktualizację Q-learning wewnątrz pętli.
    - Dla reached_goal == True wartość przyszła wynosi 0 (target = reward).
    - W pozostałych przejściach:
          target = reward + gamma * max_a' Q[next_state, a']
    - Aktualizacja:
          Q[s] += alpha * (target - Q[s])
    """
    Q = create_q_table()
    episode_rewards = []

    for ep in range(episodes):
        state = reset_env()
        total_reward = 0.0

        for step_idx in range(MAX_STEPS):
            action = choose_action(Q, state, epsilon)
            next_state, reward, reached_goal = step(state, action)
            total_reward += reward

            # TODO 3: oblicz target i zaktualizuj Q[state[0], state[1], action]
            # PODPOWIEDŹ:
            # if reached_goal:
            #     target = reward
            # else:
            #     target = reward + gamma * np.max(Q[next_state[0], next_state[1], :])
            # Q[state[0], state[1], action] += alpha * (target - Q[state[0], state[1], action])
            raise NotImplementedError("TODO 3: aktualizacja Q-learning")

            # Po uzupełnieniu TODO 3 usuń `raise` powyżej i zostaw poniższe linie:
            state = next_state
            if reached_goal:
                break

        episode_rewards.append(total_reward)

    return Q, episode_rewards


# --- KROK 3 (55–75 min): ewaluacja na 20 pozycjach testowych ---

def evaluate_agent(Q, test_starts, is_random=False):
    """
    TODO 4: Przetestuj agenta na liście test_starts (bez uczenia!).
    - Wyuczony agent: epsilon = 0.0, BEZ aktualizacji tablicy Q.
    - Agent losowy (is_random=True): losuj akcje.
    - Nie pomijaj niepowodzeń w success_rate.

    Zwróć słownik:
      success_rate, success_count, avg_steps_success, results
    gdzie results to lista słowników:
      {start, success, steps, path}
    """
    # UZUPEŁNIJ TUTAJ:
    raise NotImplementedError("TODO 4: evaluate_agent")


# =====================================================================
# 3. EKSPERYMENT (uruchamia się po uzupełnieniu TODO)
# =====================================================================

def plot_rewards(rewards, filename="wykres_nagrody.png", title="Przebieg uczenia"):
    """Wykres sumy nagród w kolejnych epizodach (+ średnia krocząca)."""
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.35, color="gray", label="Nagroda w epizodzie")
    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(rewards)),
            moving_avg,
            color="crimson",
            linewidth=2,
            label=f"Średnia krocząca ({window} ep.)",
        )
    plt.xlabel("Epizod")
    plt.ylabel("Suma nagród")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Zapisano: {filename}")
    plt.show()


def run_basic_experiment():
    """Zakres podstawowy: trening eps=0.2, porównanie z agentem losowym, ścieżka."""
    np.random.seed(SEED)

    print("=" * 60)
    print("ZAKRES PODSTAWOWY")
    print("=" * 60)

    print("\n[1/4] Trening Q-learning (300 ep., alpha=0.1, gamma=0.95, eps=0.2)...")
    Q, rewards = train_q_learning(episodes=300, alpha=0.1, gamma=0.95, epsilon=0.2)
    np.save("q_table.npy", Q)
    print("Zapisano: q_table.npy")
    plot_rewards(rewards, filename="wykres_nagrody.png", title="Uczenie Q-learning (eps=0.2)")

    print("\n[2/4] Ewaluacja agenta losowego na 20 startach...")
    eval_random = evaluate_agent(Q, TEST_STARTS, is_random=True)

    print("\n[3/4] Ewaluacja agenta wyuczonego (eps=0, bez aktualizacji Q)...")
    eval_trained = evaluate_agent(Q, TEST_STARTS, is_random=False)

    print("\n--- Wyniki (20 pozycji testowych) ---")
    print(
        f"Agent losowy:   {eval_random['success_rate']:.1f}% "
        f"({eval_random['success_count']}/20), "
        f"śr. kroków (udane): {eval_random['avg_steps_success']:.2f}"
    )
    print(
        f"Agent wyuczony: {eval_trained['success_rate']:.1f}% "
        f"({eval_trained['success_count']}/20), "
        f"śr. kroków (udane): {eval_trained['avg_steps_success']:.2f}"
    )

    print("\n[4/4] Przykładowa ścieżka ze startu (0, 0)...")
    example = eval_trained["results"][0]
    print(f"Sukces={example['success']}, kroki={example['steps']}, ścieżka={example['path']}")
    draw_path(example["path"], title="Wyuczona ścieżka z (0, 0)", filename="sciezka_wyuczona.png")

    return Q, rewards, eval_random, eval_trained


def run_extension_experiment():
    """
    Zakres dodatkowy (fakultatywnie): porównanie eps=0.05 oraz eps=0.3
    przy tej samej liczbie epizodów, parametrach i liście TEST_STARTS.
    """
    print("\n" + "=" * 60)
    print("ZAKRES DODATKOWY (eps=0.05 vs eps=0.30)")
    print("=" * 60)

    np.random.seed(SEED)
    Q_low, rewards_low = train_q_learning(episodes=300, alpha=0.1, gamma=0.95, epsilon=0.05)
    eval_low = evaluate_agent(Q_low, TEST_STARTS, is_random=False)

    np.random.seed(SEED)
    Q_high, rewards_high = train_q_learning(episodes=300, alpha=0.1, gamma=0.95, epsilon=0.30)
    eval_high = evaluate_agent(Q_high, TEST_STARTS, is_random=False)

    print(
        f"eps=0.05 -> skuteczność {eval_low['success_rate']:.1f}%, "
        f"śr. kroków {eval_low['avg_steps_success']:.2f}"
    )
    print(
        f"eps=0.30 -> skuteczność {eval_high['success_rate']:.1f}%, "
        f"śr. kroków {eval_high['avg_steps_success']:.2f}"
    )
    plot_rewards(rewards_low, filename="wykres_nagrody_eps005.png", title="Uczenie (eps=0.05)")
    plot_rewards(rewards_high, filename="wykres_nagrody_eps030.png", title="Uczenie (eps=0.30)")


if __name__ == "__main__":
    print("Plansza (0=wolne, 1=ściana, 2=cel):")
    print(BOARD)
    print(f"Liczba dozwolonych startów: {len(VALID_STARTS)}")
    print(f"Liczba pozycji testowych: {len(TEST_STARTS)}")

    print("\n--- Demo agenta losowego (0–10 min) ---")
    np.random.seed(SEED)
    demo_random_agent()

    print("\n--- Eksperyment (wymaga uzupełnienia TODO 1–4) ---")
    try:
        run_basic_experiment()
        # Zakres dodatkowy – odkomentuj po ukończeniu części podstawowej:
        # run_extension_experiment()
    except NotImplementedError as err:
        print(f"\nTODO jeszcze nieuzupełnione: {err}")
        print("Uzupełnij funkcje create_q_table / choose_action / train_q_learning / evaluate_agent")
        print("i uruchom ponownie: python warsztat2_szkielet.py")
    else:
        print("\n--- Gdy wszystko działa (wykresy + ścieżka OK) ---")
        print("Zakres dodatkowy: możesz zmienić BOARD np. na 7×7 (przykład w komentarzu przy BOARD),")
        print("ponownie uruchomić trening/test, zapisać osobne PNG/NPY i porównać różnice")
        print("w Szablon_sprawozdania.md (sekcja 6). Weź pod uwagę: osiągalność celu,")
        print("więcej epizodów / inny epsilon / MAX_STEPS, rozmiar tablicy Q.")
