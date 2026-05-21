# Projekt z kursu Optymalizacja systemów i sieci informatycznych nowej generacji

Kurs realizowany na kierunku Informatyka Stosowana, studia II stopnia, w semestrze letnim 2025/2026 na Politechnice Wrocławskiej.

Skład grupy projektowej:
- Dawid Nowak
- Jan Narożny
- Adrian Czubaty

**Cel projektu**
1. Zrozumieć model matematyczny z artykułu.
2. Zastosować solver do wyznaczania rozwiązania dla zadanego modelu.
3. Zaimplementować algorytm opisany w artykule.
4. Porównać wyniki solvera i algorytmu heurystycznego.
5. Opisać całość w sprawozdaniu końcowym.

**Struktura projektu**
- `projekt/data/` - pliki JSON z danymi wejściowymi
- `projekt/can_cplex.py` - rozwiązanie z użyciem CPLEX i aproksymacji PWL
- `projekt/can_heuristic.py` - heurystyka dwupoziomowa zgodna z sekcją 3 artykułu
- `projekt/ModelReader.py` - wczytywanie i walidacja danych wejściowych
- `projekt/routing_viz.py` - wizualizacja tensora routingu

**Środowisko**
Projekt używa `uv` oraz zależności zdefiniowanych w `pyproject.toml`.

Instalacja zależności:

```bash
uv sync
```

**Uruchamianie solvera CPLEX**
Podstawowe uruchomienie:

```bash
uv run python projekt/can_cplex.py projekt/data/data1.json
```

Przydatne opcje:
- `--tau3` - stałe opóźnienie na link
- `--rate-eps` - minimalna dodatnia szybkość dla aktywnego przepływu
- `--queue-eps` - minimalny zapas pojemności linku
- `--log-output` - pokazuje surowy log CPLEX

Przykład:

```bash
uv run python projekt/can_cplex.py projekt/data/data1.json --log-output
```

**Uruchamianie heurystyki**
Podstawowe uruchomienie:

```bash
uv run python projekt/can_heuristic.py projekt/data/data1.json
```

Przydatne opcje:
- `--tau3` - stałe opóźnienie na link
- `--rate-eps` - minimalna szybkość aktywnego przepływu
- `--queue-eps` - minimalny zapas pojemności linku
- `--max-outer-iter` - maksymalna liczba iteracji zewnętrznych heurystyki
- `--verbose` - dodatkowe komunikaty diagnostyczne

Przykład:

```bash
uv run python projekt/can_heuristic.py projekt/data/data0.json --max-outer-iter 20 --verbose
```

**Wizualizacja routingu**
Podstawowe uruchomienie:

```bash
uv run python projekt/routing_viz.py projekt/data/data1.json
```

Zapis do pliku bez otwierania okna:

```bash
uv run python projekt/routing_viz.py projekt/data/data1.json --save routing_data1.png --no-show
```

**Użycie z poziomu Pythona**
Każdy z głównych modułów można też importować:

```python
from projekt.can_cplex import solve_file as solve_cplex
from projekt.can_heuristic import solve_file as solve_heuristic

result_cplex = solve_cplex("projekt/data/data1.json")
result_heur = solve_heuristic("projekt/data/data1.json")
```

Wizualizacja:

```python
from projekt.routing_viz import visualize_file

visualize_file("projekt/data/data1.json", show=False, save_path="routing.png")
```

**Uwagi**
- `can_cplex.py` optymalizuje model z użyciem aproksymacji PWL, więc wynik CPLEX nie jest dokładnym optimum oryginalnego nieliniowego problemu.
- `can_heuristic.py` implementuje heurystykę z sekcji 3 artykułu, więc wynik również nie musi być optimum globalnym.
- Porównanie metod powinno być wykonywane na tych samych instancjach wejściowych z folderu `projekt/data/`.
