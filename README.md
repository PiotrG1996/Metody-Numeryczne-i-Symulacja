# Laboratorium MNiS
## Laboratorium 1 - Wprowadzenie w języku Python

Ten dokument zawiera rozwiązania zadań przy użyciu Pythona i biblioteki NumPy. Każde zadanie jest wyjaśnione w komentarzach i zilustrowane przykładowym kodem.

### 1. Tworzenie katalogu
Aby utworzyć katalog w Pythonie, użyj funkcji os.mkdir(). Możesz również sprawdzić, czy katalog już istnieje, aby uniknąć błędów.

```python
import os

# Tworzenie katalogu
def create_directory(directory_name):
    if not os.path.exists(directory_name):  # Sprawdza, czy katalog już istnieje
        os.mkdir(directory_name)
        print(f"Katalog '{directory_name}' został utworzony.")
    else:
        print(f"Katalog '{directory_name}' już istnieje.")

create_directory("nazwa_katalogu")  # Tworzy katalog o nazwie "nazwa_katalogu"
```

### 2. Zmiana bieżącego katalogu
Aby zmienić bieżący katalog roboczy, użyj funkcji os.chdir(). Pamiętaj, że zmiana katalogu roboczego wpływa na wszystkie operacje wykonywane w skrypcie po jej zastosowaniu.
```python
import os

# Zmiana bieżącego katalogu
def change_directory(path):
    try:
        os.chdir(path)  # Zmienia bieżący katalog na "path"
        print(f"Zmieniono katalog roboczy na: {os.getcwd()}")  # Wyświetla aktualny katalog roboczy
    except FileNotFoundError:
        print(f"Ścieżka {path} nie istnieje.")

change_directory("/path/to/directory")  # Przykład zmiany katalogu roboczego

```

### 3. Lista zawartości pustego katalogu
Funkcja `os.listdir()` zwraca zawartość katalogu. Jeśli katalog jest pusty, zwraca pustą listę.

```python
import os
print(os.listdir())  # Wynik: [] (pusta lista)
```

### 4. Zapisywanie i wczytywanie zmiennych
Aby zapisać zmienne do pliku i je ponownie wczytać, użyj `numpy.save()` i `numpy.load()`.

```python
import numpy as np

# Zapisywanie zmiennych
a = 10
b = 20
np.save("dane.npy", np.array([a, b]))  # Zapisuje zmienne do "dane.npy"

# Wczytywanie zmiennych
loaded_data = np.load("dane.npy")  # Wczytuje zmienne z "dane.npy"
print(loaded_data)  # Wynik: [10 20]
```

### 5. Czyszczenie zmiennych i konsoli
Aby usunąć zmienną z pamięci, użyj `del`.
Aby wyczyścić konsolę, użyj `os.system()`.

```python
a = 10
del a  # Usuwa zmienną 'a' z pamięci

import os
os.system('cls' if os.name == 'nt' else 'clear')  # Czyści konsolę
```

### 6. Wyświetlanie nazw zmiennych w pamięci
Aby wyświetlić wszystkie nazwy zmiennych w pamięci, użyj funkcji `globals()`.

```python
a = 10
b = 20
print([var for var in globals() if not var.startswith("__")])  # Wynik: ['a', 'b']
```

### 7. Określanie rozmiaru zmiennej
Aby określić rozmiar tablicy NumPy, użyj atrybutu `.shape`.

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)  # Wynik: (2, 3)
```

### 8. Tworzenie macierzy 3x3
Aby utworzyć macierz 3x3, użyj funkcji `numpy.array()`.

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)
```

### 9. Użycie operatora zakresu
Funkcja `numpy.arange()` generuje sekwencję liczb.

Z krokiem 1:

```python
import numpy as np
print(np.arange(1, 6))  # Wynik: [1 2 3 4 5]
```

Z krokiem 0.25:

```python
print(np.arange(0, 1.25, 0.25))  # Wynik: [0.   0.25 0.5  0.75 1.  ]
```

### 10. Dostęp do elementów macierzy
Możesz uzyskać dostęp do wierszy, kolumn lub poszczególnych elementów macierzy.

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Pierwszy wiersz
print(a[0, :])  # Wynik: [1 2 3]

# Druga kolumna
print(a[:, 1])  # Wynik: [2 5 8]

# Element (2,3)
print(a[1, 2])  # Wynik: 6
```

### 11. Wyświetlanie elementów macierzy mniejszych niż wartość
Aby wyświetlić elementy macierzy mniejsze niż określona wartość, użyj indeksowania logicznego.

```python
print(a[a < 5])  # Wynik: [1 2 3 4]
```

### 12. Generowanie macierzy jednostkowej 5x5
Użyj `numpy.eye()`.

```python
import numpy as np
print(np.eye(5))  # Wynik: macierz jednostkowa 5x5
```

### 13. Generowanie macierzy zerowej 3x3
Użyj `numpy.zeros()`, aby utworzyć macierz zerową.

```python
print(np.zeros((3, 3)))  # Wynik: macierz zerowa 3x3
```

### 14. Generowanie macierzy jedynek 3x3
Użyj ```numpy.ones()```, aby utworzyć macierz wypełnioną jedynkami.

```python
print(np.ones((3, 3)))  # Wynik: macierz 3x3 wypełniona jedynkami
```

### 15. Generowanie macierzy diagonalnej 3x3
Użyj ```numpy.diag()```, aby utworzyć macierz diagonalną.

```python
print(np.diag([1, 2, 3]))  # Wynik: macierz diagonalna 3x3 z wartościami [1, 2, 3]
```

### 16. Generowanie losowej macierzy 3x3
Użyj ```numpy.random.rand()```, aby utworzyć macierz z losowymi wartościami.

```python
print(np.random.rand(3, 3))  # Wynik: macierz 3x3 z losowymi wartościami między 0 a 1
```

### 17. Pobieranie głównej przekątnej macierzy
Użyj ```numpy.diag()```, aby pobrać główną przekątną macierzy.

```python
a = np.random.rand(3, 3)
print(np.diag(a))  # Wynik: główna przekątna macierzy 'a'
```

### 18. Łączenie macierzy
Aby połączyć dwie macierze 2x2 poziomo, użyj ```numpy.hstack()```.

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.hstack((a, b))  # Poziome łączenie macierzy
print(c)
```

### 19. Macierze trójkątne górne i dolne
Użyj ```numpy.triu()```, aby uzyskać górną część macierzy trójkątnej.
Użyj ```numpy.tril()```, aby uzyskać dolną część macierzy trójkątnej.

```python
print(np.triu(a))  # Górna część macierzy trójkątnej
print(np.tril(a))  # Dolna część macierzy trójkątnej
```

### 20. Transponowanie macierzy
Użyj atrybutu ```.T```, aby transponować macierz.

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a.T  # Transponowanie macierzy
print(b)
```

# Laboratorium 2 - Układy równań
## 1. Tworzenie kopii macierzy `a`
```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a.copy()  # Tworzenie kopii macierzy
print(b)
```

## 2. Zmiana rozmiaru macierzy `a` na (2x2) bez utraty danych
```python
a_resized = a[:2, :2]  # Wycinanie części macierzy
print(a_resized)
```

## 3. Powiększenie macierzy `a` o dwa wiersze zer
```python
a_expanded = np.vstack((a, np.zeros((2, 3))))
print(a_expanded)
```

## 4. Definicja pustej macierzy
```python
a_empty = np.empty((0, 0))
print(a_empty)
```

## 5. Tworzenie pustej macierzy i dodawanie do niej wierszy
```python
a = np.empty((0, 3))
a = np.vstack([a, [1, 2, 3]])
a = np.vstack([a, [4, 5, 6]])
a = np.vstack([a, [7, 8, 9]])
print(a)
```

## 6. Indeksowanie elementów macierzy
```python
print(a[1, 1])  # Indeksowanie standardowe
print(a[1][1])  # Alternatywny sposób indeksowania
```

## 7. Zmiana rozmiaru macierzy `a` na (2x6) przy użyciu `reshape`
```python
a_reshaped = a.reshape(2, 6)
print(a_reshaped)
```

## 8. Przywrócenie macierzy `a` do oryginalnego rozmiaru (3x3)
```python
a_restored = a.reshape(3, 3)
print(a_restored)
```

## 9. Wyświetlanie macierzy w postaci kolumnowej
```python
print(a.flatten(order='F'))
```

## 10. Tworzenie magicznego kwadratu i sprawdzanie sum
```python
a = np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])
print(np.sum(a, axis=0))  # Suma kolumn
print(np.sum(a, axis=1))  # Suma wierszy
```

## 11. Generowanie macierzy Hilberta dla `n=4`
```python
from scipy.linalg import hilbert

h = hilbert(4)
print(h)
```

## 12. Podstawowe operacje na macierzach
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[1, -1], [-1, 1]])

print(a + b)  # Dodawanie
print(a - b)  # Odejmowanie
print(a * b)  # Mnożenie element po elemencie
print(a / 2)  # Dzielenie przez skalar
```

## 13. Dzielenie macierzowe (lewostronne i prawostronne)
```python
a = np.array([[1, 2, 3]])
b = np.array([[6, 5, 4]])

x_left = np.linalg.lstsq(a.T, b.T, rcond=None)[0]  # Lewostronne
x_right = np.linalg.lstsq(b, a, rcond=None)[0]  # Prawostronne
print(x_left, x_right)
```

## 14. Rozwiązywanie układu równań liniowych
```python
A = np.array([[2, -2, 1], [1, 4, -2], [6, -1, -1]])
b = np.array([-4, 1, 2])
x = np.linalg.solve(A, b)
print(x)
```

## 15. Rozwiązywanie równań kwadratowych
```python
import sympy as sp

a, b, c = sp.symbols('a b c')
x = sp.symbols('x')
quadratic_eq = sp.Eq(a*x**2 + b*x + c, 0)
solutions = sp.solve(quadratic_eq, x)
print(solutions)
```

## 16. Rozwiązywanie układu równań z macierzą 4x4
```python
import math

# Input coefficients
a = float(input("Enter coefficient a: "))
b = float(input("Enter coefficient b: "))
c = float(input("Enter coefficient c: "))

# Calculating discriminant
D = b**2 - 4*a*c

if D > 0:
    x1 = (-b + math.sqrt(D)) / (2*a)
    x2 = (-b - math.sqrt(D)) / (2*a)
    print(f"Two real solutions: x1 = {x1}, x2 = {x2}")
elif D == 0:
    x = -b / (2*a)
    print(f"One real solution: x = {x}")
else:
    real_part = -b / (2*a)
    imaginary_part = math.sqrt(-D) / (2*a)
    print(f"Complex solutions: x1 = {real_part} + {imaginary_part}i, x2 = {real_part} - {imaginary_part}i")
```

### 17.
```python
import numpy as np

# Coefficient matrix A
A = np.array([[1, 1, 1, 1],
              [3, 2, 4, 5],
              [2, 1, -1, -2],
              [4, 3, 2, 1]])

# Constants vector B
B = np.array([2, -1, -1, 1])

# Calculating the determinant of A
det_A = np.linalg.det(A)

if det_A != 0:
    # Cramer's Rule to solve for x, y, z, a
    X = np.linalg.inv(A).dot(B)
    print(f"Solution: x = {X[0]}, y = {X[1]}, z = {X[2]}, a = {X[3]}")
else:
    print("No unique solution exists (determinant is zero).")
```

### 18.
```python
import numpy as np

# Input coefficients
a = float(input("Enter coefficient a: "))
b = float(input("Enter coefficient b: "))
c = float(input("Enter coefficient c: "))
d = float(input("Enter coefficient d: "))

# Defining the cubic equation
coefficients = [a, b, c, d]

# Finding the roots using numpy's roots function
roots = np.roots(coefficients)

# Printing the roots
print("The roots of the cubic equation are:")
for root in roots:
    print(root)
```

# Laboratorium 3 - Analiza sygnałów zaszumionych

## Wprowadzenie

W ćwiczeniu opracowany został program, który na podstawie plików wejściowych:
1. **arguments of sine.txt** - sygnał wejściowy wymuszenia
2. **sine.txt** - odpowiedź w formie funkcji sinusoidalnej
3. **sine with noise.txt** - zaszumiony sygnał wyjściowy

Program analizuje sygnał wzorcowy oraz zaszumiony, obliczając błędy, wartość skuteczną, odchylenie standardowe oraz błędy średnie.

## Zadania

### Zadanie 1 - Określenie błędów dla poszczególnych próbek sygnału

Błąd dla każdej próbki sygnału obliczany jest jako różnica między wartością zaszumionego sygnału a wartością sygnału wzorcowego:

\[
\text{Błąd}_i = y_{\text{zaszumiony}}(i) - y_{\text{wzorcowy}}(i)
\]

W celu obliczenia błędów dla poszczególnych próbek sygnału wczytano dane z plików **sine.txt** i **sine with noise.txt**, a następnie obliczono różnice.

```python
import numpy as np

# Wczytanie danych z plików
sine_values = np.loadtxt('sine.txt')
noise_values = np.loadtxt('sine with noise.txt')

# Obliczanie błędu dla każdej próbki
errors = noise_values - sine_values

# Wyświetlenie wyników
print("Błędy dla poszczególnych próbek sygnału:")
print(errors)
```

### Zadanie 2 - Określ jaki jest błąd średni (względny, bezwzględny dla sygnału zaszumionego i wzorcowego) oraz jak on wpływa na kształt zaszumionego sygnału (obliczyć nowe wartości próbek zaszumionych z uwzględnieniem błędu średniego).

![Przykład](images/blad_sredni.png)

```python

# Obliczanie błędu średniego bezwzględnego
mean_absolute_error = np.mean(np.abs(errors))

# Obliczanie błędu średniego względnego
mean_relative_error = np.mean(np.abs(errors / sine_values))

print(f"Błąd średni bezwzględny: {mean_absolute_error}")
print(f"Błąd średni względny: {mean_relative_error}")

# Obliczanie nowych próbek zaszumionych uwzględniając błąd średni
corrected_noise_values = noise_values - mean_absolute_error
print("Skorygowane wartości próbek zaszumionych:")
print(corrected_noise_values)
```

### 3. Określenie wartości skutecznej sygnałów

![Przykład](images/wartosc_skuteczna.png)

```python

# Obliczanie wartości skutecznej dla sygnałów
effective_value_sine = np.sqrt(np.mean(sine_values**2))
effective_value_noise = np.sqrt(np.mean(noise_values**2))

print(f"Wartość skuteczna sygnału sine.txt: {effective_value_sine}")
print(f"Wartość skuteczna sygnału sine with noise.txt: {effective_value_noise}")
```

### 4. Określ jakie jest odchylenie standardowe dla zaszumionego sygnału, porównaj z sygnałem, w którym wprowadzono korektę w postaci błędu średniego.

![Przykład](images/odchylenie.png)


```python
# Obliczanie odchylenia standardowego dla zaszumionego sygnału
std_noise = np.std(noise_values)

# Obliczanie odchylenia standardowego dla skorygowanego sygnału
std_corrected_noise = np.std(corrected_noise_values)

print(f"Odchylenie standardowe dla zaszumionego sygnału: {std_noise}")
print(f"Odchylenie standardowe dla skorygowanego sygnału: {std_corrected_noise}")
```

### 5. Rozkład błędów wokół wartości średniej Rozkład błędów wokół średniej możemy zwizualizować przy pomocy histogramu. Histogram przedstawia rozkład błędów w obrębie wartości średniej.

```python
import matplotlib.pyplot as plt

# Rysowanie histogramu błędów
plt.hist(errors, bins=30, edgecolor='black')
plt.title('Rozkład błędów wokół wartości średniej')
plt.xlabel('Błąd')
plt.ylabel('Liczba próbek')
plt.show()
```

### 6. Określ ile wynosi błąd średniej arytmetycznej
```python
# Obliczanie błędu średniej arytmetycznej
mean_noise = np.mean(noise_values)
mean_sine = np.mean(sine_values)

mean_error = mean_noise - mean_sine
print(f"Błąd średniej arytmetycznej: {mean_error}")
```

# Laboratorium 4 - Rozwiązywanie równań nieliniowych


## 3.1 Metoda Bisekcji

## Przykład: Pierwiastek Trzeciego Stopnia

Problem obliczenia pierwiastka trzeciego stopnia z liczby \( a \) można sprowadzić do znalezienia pierwiastka równania:

$$ f(x) = x^3 - a $$

### Implementacja funkcji w Pythonie:

```python
import numpy as np
import matplotlib.pyplot as plt

# Funkcja definiująca równanie
def zadanie(x, z):
    return x**3 - z

# Funkcja do podziału przedziałów
def szukanie(gr_l, gr_p):
    return gr_l + (gr_p - gr_l) / 2

# Inicjalizacja
gr_l = -2
gr_p = 2
z = 3  # Szukamy pierwiastka z 3
delta = 10**-5
t = np.linspace(gr_l, gr_p, 1000)

# Metoda bisekcji
n = 1
tn = [n - 1]
x = [gr_p]
gr_c = gr_p

while abs(zadanie(gr_c, z)) > delta and abs(gr_p - gr_l) > delta:
    n += 1
    tn.append(n - 1)
    gr_c = szukanie(gr_l, gr_p)
    yc = zadanie(gr_c, z)
    ya = zadanie(gr_l, z)
    zero = ya * yc
    
    if zero < 0:
        gr_p = gr_c
    else:
        gr_l = gr_c

    x.append(gr_c)

# Wykres iteracji
plt.subplot(121)
plt.plot(t, zadanie(t, z))
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.title('Wykres funkcji')

plt.subplot(122)
plt.plot(tn, x)
plt.title('Zbieżność metody bisekcji')

plt.show()
```

### 1: Zaimplementowanie algorytmu obliczania miejsc zerowych dla funkcji:
$$ \( f(x) = x^2 - 2 \) $$ 
### w przedziale [0, 3].

```python
import numpy as np

def f(x):
    """Funkcja, dla której szukamy pierwiastka: f(x) = x^2 - 2"""
    return x**2 - 2

def bisection_method(f, a, b, delta):
    """
    Implementacja metody bisekcji do znalezienia pierwiastka funkcji f w przedziale [a, b].
    """
    iterations = []
    while abs(b - a) > delta:
        c = (a + b) / 2
        iterations.append([a, b, c])
        if f(c) == 0:
            return c, np.array(iterations)  
        elif f(a) * f(c) < 0:
            b = c  
        else:
            a = c  
    return (a + b) / 2, np.array(iterations)

# Definicja parametrów
a, b, delta = 0, 3, 1e-5
root, iterations = bisection_method(f, a, b, delta)
print(f"Pierwiastek: {root}")

# Tabela zostanie wygenerowana za pomocą numpy.array, która zawiera iteracyjne wartości 
print("Tabela iteracji:")
print(iterations)
```

### 2: Przedstaw w postaci tabeli kolejne kroki przybliżeń jakie zostały otrzymane.

```python
import pandas as pd

pd.DataFrame(iterations, columns=["a", "b", "c"])
```

### 3: Testowanie algorytmu dla różnych wartości dokładności (delta)

```python
# Możemy zmieniać wartość delta i obserwować wpływ na liczbę iteracji:
# Testowanie dla różnych wartości delta

deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
for delta in deltas:
    root, iterations = bisection_method(f, a, b, delta)
    print(f"Pierwiastek dla delta={delta}: {root}, Liczba iteracji: {iterations.shape[0]}")
```

### 4: Jak zmienia się liczba iteracji w zależności od delta
Liczba iteracji będzie maleć w miarę zmniejszania się wartości delta, ponieważ dokładność rozwiązania będzie coraz wyższa.

### 5: Modyfikacja skryptu do wyznaczania miejsc zerowych w przedziale [-3, 3]
Aby znaleźć pierwiastki w przedziale [-3, 3], wystarczy zmienić przedziały na:

```python
a, b = -3, 3
root, iterations = bisection_method(f, a, b, delta)

# 3.2 Metoda Newtona
### 1: Implementacja algorytmu obliczania pierwiastka trzeciego stopnia

```python
def f_cubic(x, a):
    """Funkcja, dla której szukamy pierwiastka trzeciego stopnia: f(x) = x^3 - a"""
    return x**3 - a

def f_cubic_derivative(x):
    """Pochodna funkcji f_cubic(x, a) względem x"""
    return 3 * x**2

def newton_method(f, f_prime, x0, delta, max_iter=1000):
    """Metoda Newtona do znalezienia pierwiastka funkcji f."""
    iterations = []
    x = x0
    for i in range(max_iter):
        x_new = x - f(x) / f_prime(x)
        iterations.append([i, x, x_new])
        if abs(x_new - x) < delta:
            return x_new, np.array(iterations)
        x = x_new
    return x, np.array(iterations)
```

### 2: Wybór metody określania dokładności obliczeń
Użyjemy warunku zakończenia: 

```python
import numpy as np
import matplotlib.pyplot as plt

        # Warunek końcowy: sprawdzamy, czy wartość funkcji jest wystarczająco bliska zeru
        if abs(fx) <= epsilon:
            break  # Zakończenie obliczeń, gdy funkcja jest bliska zeru
        
        x = x_new  # Przechodzimy do nowego przybliżenia
    
    return x_values

# Wybór liczby podpierwiastkowej (np. pierwiastek trzeciego stopnia z 27)
a = 27
epsilon = 1e-6
max_iter = 100

# Punkt startowy (możesz dowolnie zmienić)
x0 = 5

# Obliczanie przybliżeń
x_values = newton_cubic_root(a, epsilon, max_iter, x0)
```

### 3: Modyfikacja programu dla różnych punktów startowych
Aby zrealizować ten punkt, generujemy różne wartości początkowe 

```python
import matplotlib.pyplot as plt

chosen_number = 27
random_x0s = np.random.randint(1, chosen_number, 4)
plt.figure(figsize=(10, 6))
for x0 in random_x0s:
    _, iterations = newton_method(lambda x: f_cubic(x, chosen_number), f_cubic_derivative, x0, 1e-5)
    iterations = np.array(iterations)
    plt.plot(iterations[:, 0], iterations[:, 2], label=f"x0 = {x0}")

plt.xlabel("Iteracja")
plt.ylabel("xk")
plt.legend()
plt.title(f"Dążenie algorytmu Newtona do wyniku dla liczby {chosen_number}")
plt.show()
```

### 4: wylosuj cztery różne punkty startowe i wykreśl zależność pokazującą dążenie algorytmu do wyniku (xk = f(iteracja))

```python
import numpy as np
import matplotlib.pyplot as plt

# Funkcja obliczająca pierwiastek trzeciego stopnia
def newton_cubic_root(a, epsilon=1e-6, max_iter=100, x0=None):
    # Funkcja f(x) = x^3 - a
    # Pochodna f'(x) = 3 * x^2
    if x0 is None:
        x0 = a / 2  # Startowy punkt (można zmienić, ale stały dla wszystkich przypadków)
    
    x = x0
    iterations = 0
    x_values = [x]  # Lista przechowująca wartości x w każdej iteracji
    
    while iterations < max_iter:
        fx = x**3 - a
        fpx = 3 * x**2
        x_new = x - fx / fpx
        x_values.append(x_new)
        iterations += 1
        if abs(x_new - x) < epsilon:
            break
        x = x_new
        
    return x_values

# Wybór liczby podpierwiastkowej (np. pierwiastek trzeciego stopnia z 27)
a = 27
epsilon = 1e-6
max_iter = 100

# Losowanie czterech różnych punktów startowych
np.random.seed(42)  # Ustawiamy ziarno dla powtarzalności wyników
start_points = np.random.uniform(1, 10, 4)  # Losujemy 4 różne punkty startowe w zakresie [1, 10]

# Tworzymy wykres
plt.figure(figsize=(10, 6))

# Dla każdego punktu startowego obliczamy iteracje i rysujemy wykres
for x0 in start_points:
    x_values = newton_cubic_root(a, epsilon, max_iter, x0)
    plt.plot(range(len(x_values)), x_values, label=f'Start: {x0:.2f}')

# Dodajemy legendę, tytuł, etykiety i siatkę
plt.title(f"Zbieżność algorytmu Newtona dla pierwiastka 3 stopnia z {a}")
plt.xlabel("Iteracja")
plt.ylabel("Wartość x (przybliżenie pierwiastka)")
plt.legend(title="Punkty startowe")
plt.grid(True)

# Wyświetlamy wykres
plt.show()
```

### 5: Obliczenie pierwiastków 3 stopnia dla liczb w zakresie od 1 do 100000

```python

import numpy as np
import matplotlib.pyplot as plt

# Funkcja obliczająca pierwiastek trzeciego stopnia
def newton_cubic_root(a, epsilon=1e-6, max_iter=100):
    # Funkcja f(x) = x^3 - a
    # Pochodna f'(x) = 3x^2
    x = a / 2  # Startowy punkt (można zmienić, ale stały dla wszystkich przypadków)
    iterations = 0
    while iterations < max_iter:
        fx = x**3 - a
        fpx = 3 * x**2
        x_new = x - fx / fpx
        iterations += 1
        if abs(x_new - x) < epsilon:
            break
        x = x_new
    return x, iterations

# Zakres liczb, dla których obliczamy pierwiastki (od 1 do 100000)
numbers = np.arange(1, 100001)
iterations_list = []

# Oblicz pierwiastki i liczbę iteracji
for num in numbers:
    _, iterations = newton_cubic_root(num)
    iterations_list.append(iterations)

# Wykres liczby iteracji w zależności od liczby podpierwiastkowej
plt.figure(figsize=(10, 6))
plt.plot(numbers[:1000], iterations_list[:1000], marker='o', linestyle='-', color='b')  # Wyświetlamy tylko pierwsze 1000 punktów dla lepszej wizualizacji
plt.title("Liczba iteracji algorytmu Newtona w zależności od liczby podpierwiastkowej (1 do 100000)")
plt.xlabel("Liczba podpierwiastkowa (a)")
plt.ylabel("Liczba iteracji")
plt.grid(True)
plt.show()

# Korelacja Pearsona między liczbą podpierwiastkową a liczbą iteracji
correlation = np.corrcoef(numbers, iterations_list)
print(f"Współczynnik korelacji Pearsona: {correlation[0, 1]}")

```

### 5: Korelacja między liczbą podpierwiastkową a ilością iteracji
Na wykresie widać, jak liczba iteracji zmienia się w zależności od liczby podpierwiastkowej. Wiele razy liczba iteracji nie zależy w prosty sposób od wartości 
𝑎
a, ale może być różna w zależności od punktu startowego i zbieżności metody Newtona.


# Laboratorium 5 - Interpolacja wielomianowa

### 1. Zaimplementowanie algorytmu interpolacji funkcji metodą Newtona

```python

import numpy as np

def newton_interpolation(x_points, y_points, x):
    n = len(x_points)
    # Obliczanie różnic dzielonych
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y_points

    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / (x_points[i + j] - x_points[i])

    # Obliczanie wartości wielomianu Newtona
    result = divided_diff[0, 0]
    for i in range(1, n):
        term = divided_diff[0, i]
        for j in range(i):
            term *= (x - x_points[j])
        result += term
    return result

# Przykładowe dane:
x_points = [1, 2, 4.5, 5]
y_points = [-10.5, -16.11, 11.8125, 27.5]

# Obliczanie wartości funkcji w punkcie x = 3
x_val = 3
y_val = newton_interpolation(x_points, y_points, x_val)
print(f"Interpolacja Newtona dla x = {x_val}: y = {y_val}")
```

### 2. Sprawdzenie działania algorytmu dla podanych danych (Wielomian 3-go stopnia)

```python
import matplotlib.pyplot as plt

# Wykorzystanie wcześniej zaimplementowanej funkcji Newtona
def plot_newton_interpolation(x_points, y_points, degree, x_vals):
    n = len(x_points)
    interpolated_values = [newton_interpolation(x_points, y_points, x) for x in x_vals]

    plt.plot(x_vals, interpolated_values, label=f"Interpolacja Newtona (stopień {degree})", color="blue")
    plt.scatter(x_points, y_points, color="red", label="Punkty danych")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.title(f"Interpolacja Newtona (stopień {degree})")
    plt.show()

# Wartości do wykresu
x_vals = np.linspace(min(x_points), max(x_points), 100)

# Wykres
plot_newton_interpolation(x_points, y_points, degree=3, x_vals=x_vals)
```

### 3. Obliczanie błędu interpolacji

```python
def interpolation_error(x_points, y_points, x_actual, x_val):
    # Rzeczywista wartość funkcji
    y_actual = np.interp(x_val, x_points, y_points)
    
    # Wartość funkcji interpolowanej
    y_interpolated = newton_interpolation(x_points, y_points, x_val)
    
    # Błąd
    return abs(y_actual - y_interpolated)

# Obliczanie błędu interpolacji w punkcie x = 3
error = interpolation_error(x_points, y_points, x_points, x_val)
print(f"Błąd interpolacji w punkcie x = {x_val}: {error}")
```

### 4. Interpolacja funkcji za pomocą wielomianów stopni 2-4

```python
from numpy.polynomial.polynomial import Polynomial

# Dane
x_points_2 = [-2, 0, 1, 2.5, 4]
y_points_2 = [-14, 9, 4, 9.625, 175]

# Interpolacja dla stopnia 2, 3 i 4
def polynomial_interpolation(x_points, y_points, degree):
    p = Polynomial.fit(x_points, y_points, degree)
    return p

# Obliczanie błędów dla różnych stopni wielomianu
errors = {}
for degree in [2, 3, 4]:
    p = polynomial_interpolation(x_points_2, y_points_2, degree)
    error_values = [abs(p(x) - np.interp(x, x_points_2, y_points_2)) for x in x_points_2]
    errors[degree] = (min(error_values), max(error_values))

print(errors)

# Wybór najlepszego wielomianu (z najmniejszym błędem)
best_degree = min(errors, key=lambda k: errors[k][0])
print(f"Najlepszy stopień wielomianu: {best_degree}")

# Wykres najlepszego wielomianu
p_best = polynomial_interpolation(x_points_2, y_points_2, best_degree)
x_vals_2 = np.linspace(min(x_points_2), max(x_points_2), 100)
y_vals_2 = p_best(x_vals_2)

plt.plot(x_vals_2, y_vals_2, label=f"Wielomian {best_degree} stopnia")
plt.scatter(x_points_2, y_points_2, color="red", label="Punkty danych")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.title(f"Interpolacja funkcji dla stopnia {best_degree}")
plt.show()
```

### 5. Interpolacja prędkości rakiety
```python
# Dane
t_points = [0, 10, 15, 20, 22.5, 30]
v_points = [0, 227.4, 362.8, 517.35, 602.97, 901.67]

# Interpolacja prędkości rakiety za pomocą wielomianu 3-go stopnia
p_velocity = polynomial_interpolation(t_points, v_points, 3)

# Obliczanie prędkości w t1=45s, t2=60s, t3=90s
t_values = [45, 60, 90]
v_values = [p_velocity(t) for t in t_values]

print(f"Prędkość rakiety w t1=45s: {v_values[0]}")
print(f"Prędkość rakiety w t2=60s: {v_values[1]}")
print(f"Prędkość rakiety w t3=90s: {v_values[2]}")

# Wykres
t_vals = np.linspace(min(t_points), max(t_points), 100)
v_vals = p_velocity(t_vals)

plt.plot(t_vals, v_vals, label="Interpolacja prędkości rakiety")
plt.scatter(t_points, v_points, color="red", label="Punkty danych")
plt.legend()
plt.xlabel("Czas (s)")
plt.ylabel("Prędkość (m/s)")
plt.grid(True)
plt.title("Interpolacja prędkości rakiety")
plt.show()
```

### 6. Interpolacja trajektorii ruchu robota
```python
# Dane
x_robot = [72, 71, 60, 50, 35, 50]
y_robot = [42.5, 52.5, 78.1, 92, 106, 120]

# Interpolacja trajektorii za pomocą wielomianu 5-go stopnia (dowolny stopień)
p_trajectory_x = polynomial_interpolation(range(len(x_robot)), x_robot, 5)
p_trajectory_y = polynomial_interpolation(range(len(y_robot)), y_robot, 5)

# Wykres
x_vals_robot = np.linspace(0, len(x_robot)-1, 100)
x_vals_interpolated = p_trajectory_x(x_vals_robot)
y_vals_interpolated = p_trajectory_y(x_vals_robot)

plt.plot(x_vals_interpolated, y_vals_interpolated, label="Interpolacja trajektorii")
plt.plot([0, len(x_robot)-1], [80, 80], label="Granica obszaru roboczego", color="red", linestyle="--")
plt.scatter(range(len(x_robot)), y_robot, color="green", label="Punkty danych")
plt.legend()
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title("Interpolacja trajektorii ruchu robota")
plt.grid(True)
plt.show()

```


