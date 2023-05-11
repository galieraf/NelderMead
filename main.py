"""
Module for working with the Nelder-Mead algorithm
"""
from tkinter.messagebox import showerror
import tkinter as tk
from tkinter import ttk
import re
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import nelder_mead as nm

pattern_4d = r'^[xyzasincosatanlog\d\+\-\*/\(\)\^epi]*$'
pattern_3d = r'^[xyasincosatanlog\d\+\-\*/\(\)\^epi]*$'
pattern_2d = r'^[xasincosatanlog\d\+\-\*/\(\)\^epi]*$'

functions = [nm.parabola_0, nm.parabola_1, nm.parabola_2, nm.ff_dim_2, nm.parabaloid, nm.rosenbrock,
             nm.easy_3_dim, nm.harder_3_dim]

functions_name = ['1) x^2', '2) (x-1)^2', '3) (x+3)^2+2', '4) x^3+8y^3-6xy+1', '5) x^2+y^2',
                  '6) (1-x)^2+100(y-x^2)^2', '7) x^2+y^2+z^2', '8) (x^2+y^2-4)^2+(x^2+z^2-4)^2+(y^2+z^2-4)^2']


# ----------------------------------------------------------------------------------------------------------------------
def replace_symbols_with_constants(s):
    """Replaces 'e' with Euler's number and 'pi' with Pi in a sympy-compatible string."""
    return s.replace('e', 'E').replace('pi', 'pi')


# ----------------------------------------------------------------------------------------------------------------------
def is_valid_function(function_str, dimension):
    """
    The function checks if the entered function is valid

    :param function_str: a function to check
    :param dimension: function dimension, either 2 or 3
    :return: True if is valid False if not
    """
    if dimension == 2:
        pattern = pattern_2d
    elif dimension == 3:
        pattern = pattern_3d
    else:
        pattern = pattern_4d
    function_str = function_str.replace(' ', '')
    if re.fullmatch(pattern, function_str):
        try:

            sp.sympify(function_str)
            return True
        except sp.SympifyError:
            return False
    return False


# ----------------------------------------------------------------------------------------------------------------------
def plot_2d(selected_function, initial_point, edge_length, tol):
    """
    The function for displaying a 2d function and its extremum

    :param selected_function: a function to draw
    The following parameters are used to find the extremum according to the Nelder Mead algorithm
    :param initial_point: coordinates of initial point
    :param edge_length: initial simplex edge length
    :param tol: requiring tolerance
    :return: None
    """

    point = nm.nelder_mead(selected_function, 1, initial_point, edge_length, tol)
    if point == -1:
        showerror("Error", "It seems that the function does not have an extremum. "
                           "If you are not agree try to change initial parameters")
        return

    extremum_x = point[0].get_x()[0]
    listik = np.array([extremum_x])
    extremum_y = selected_function(listik)

    x = np.linspace(extremum_x - 20, extremum_x + 20, 1000)
    y = np.array([selected_function([x_i]) for x_i in x])

    plt.plot(x, y, label="y = f(x)")
    plt.scatter(extremum_x, extremum_y, c="red", label="Extremum")

    plt.text(extremum_x, extremum_y, f'({extremum_x:.2f}, {extremum_y:.2f})')

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function and Extremum")
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
def plot_3d(selected_function, initial_point, edge_length, tol):
    """
    The function for displaying a 3d function and its extremum

    :param selected_function: a function to draw
    The following parameters are used to find the extremum according to the Nelder Mead algorithm
    :param initial_point: coordinates of initial point
    :param edge_length: initial simplex edge length
    :param tol: requiring tolerance
    :return: None
    """
    point = nm.nelder_mead(selected_function, 2, initial_point, edge_length, tol)
    if point == -1:
        showerror("Error", "It seems that the function does not have an extremum. "
                           "If you are not agree try to change initial parameters")
        return

    extremum_x = point[0].get_x()[0]
    extremum_y = point[0].get_x()[1]
    listik = [extremum_x, extremum_y]
    extremum_z = selected_function(listik)

    x = np.linspace(extremum_x - 20, extremum_x + 20, 100)
    y = np.linspace(extremum_y - 20, extremum_y + 20, 100)

    x, y = np.meshgrid(x, y)
    z = selected_function(np.array([x, y]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, alpha=0.5, cmap="viridis")
    ax.scatter(extremum_x, extremum_y, extremum_z, c="red", label="Extremum", s=100)

    ax.text(extremum_x, extremum_y, extremum_z, f'({extremum_x:.2f}, {extremum_y:.2f}, {extremum_z:.2f})')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Function and Extremum")
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
def show_4d(selected_function, initial_point, edge_length, tol):
    """
     Functions displays result for three variable

    :param selected_function: a function to draw
    The following parameters are used to find the extremum according to the Nelder Mead algorithm
    :param initial_point: coordinates of initial point
    :param edge_length: initial simplex edge length
    :param tol: requiring tolerance
    :return: None
    """
    point = nm.nelder_mead(selected_function, 3, initial_point, edge_length, tol)
    if point == -1:
        showerror("Error", "It seems that the function does not have an extremum. "
                           "If you are not agree try to change initial parameters")
        return

    extremum_x = point[0].get_x()[0]
    extremum_y = point[0].get_x()[1]
    extremum_z = point[0].get_x()[2]

    value = selected_function([extremum_x, extremum_y, extremum_z])

    def close_4d_window():
        result_window.destroy()

    result_window = tk.Toplevel(root)
    result_window.title("4D Function Result")
    result_window.geometry("400x150")

    label_extremum = tk.Label(result_window, text=f"Extremum: ({extremum_x:.4f}, {extremum_y:.4f}, {extremum_z:.4f})")
    label_value = tk.Label(result_window, text=f"Function value: {value:.4f}")

    ok_button = tk.Button(result_window, text="OK", command=close_4d_window)

    label_extremum.pack(padx=10, pady=10)
    label_value.pack(padx=10, pady=10)
    ok_button.pack(padx=10, pady=10)

    label_extremum.config(font=("Arial", BASE_FONT_SIZE))
    label_value.config(font=("Arial", BASE_FONT_SIZE))
    ok_button.config(font=("Arial", BASE_FONT_SIZE))


# ----------------------------------------------------------------------------------------------------------------------
def get_function_by_index(index):
    """
    Returns a function by its index

    :param index: a function index
    :return: function from "functions" list
    """
    return functions[index]


# ----------------------------------------------------------------------------------------------------------------------
def show_parameters_window():
    """
    Shows the main window of the program where the user can enter data

    :return: None
    """
    label_init_point_x.grid(row=0, column=1)
    entry_init_point_x.grid(row=0, column=2)
    label_init_point_y.grid(row=1, column=1)
    entry_init_point_y.grid(row=1, column=2)
    label_init_point_z.grid(row=2, column=1)
    entry_init_point_z.grid(row=2, column=2)
    label_edge_length.grid(row=3, column=1)
    entry_edge_length.grid(row=3, column=2)
    label_tol.grid(row=4, column=1)
    entry_tol.grid(row=4, column=2)
    label_function.grid(row=5, column=1)
    combo_function.grid(row=5, column=2)
    empty_label1 = tk.Label(root, text="")
    empty_label1.grid(row=6, column=0)
    label_custom_function2d.grid(row=7, column=1)
    entry_custom_function2d.grid(row=7, column=2)
    label_custom_function3d.grid(row=8, column=1)
    entry_custom_function3d.grid(row=8, column=2)
    label_custom_function4d.grid(row=9, column=1)
    entry_custom_function4d.grid(row=9, column=2)
    empty_label4 = tk.Label(root, text="")
    empty_label4.grid(row=10, column=0)
    submit_button.grid(row=11, column=1, columnspan=2)
    combo_function.current(0)


# ----------------------------------------------------------------------------------------------------------------------
def show_info_window():
    """
    Shows an information window after starting the program

    :return: None
    """

    def on_info_window_close():
        """
        Destroys the window and shows the main program window

        :return: None
        """
        info_window.destroy()
        show_parameters_window()
        root.deiconify()

    info_window = tk.Toplevel(root)
    info_window.title("Information")
    info_window.protocol("WM_DELETE_WINDOW", on_info_window_close)
    info_window.minsize(600, 320)  # Minimum size of the window
    info_window.maxsize(800, 600)  # Maximum size of the window

    # Configure the grid
    for j in range(10):
        info_window.grid_columnconfigure(j, weight=1)
        info_window.grid_rowconfigure(j, weight=1)

    # Get screen size
    scrn_width = root.winfo_screenwidth()
    scrn_height = root.winfo_screenheight()

    # Set window size as a percentage of screen size
    wndw_width = int(scrn_width * 0.5)  # 50% of screen width
    wndw_height = int(scrn_height * 0.38)  # 38% of screen height

    # Set the geometry of the window
    info_window.geometry(f"{wndw_width}x{wndw_height}")

    label_edge_length_info = tk.Label(info_window, text="1) Edge length must be in the range [1, 10].", justify="left")
    label_tol_info = tk.Label(info_window, text="2) Tolerance must be in the range [0.000001, 0.1].", justify="left")
    label_info1 = tk.Label(info_window, text='3) If you are not sure, do not change the parameters', justify="left")
    label_info2 = tk.Label(info_window, text='4) If the parameters are correct and the result is strange - '
                                             'try changing\n the initial point', justify="left")
    label_info3 = tk.Label(info_window, text='5) Starting points can be any but keep in mind that you can hit '
                                             'a local and\n not a global extremum or or one of several extremes'
                                             ' (algorithm limitation)', justify="left")
    label_info4 = tk.Label(info_window, text='Press "OK" to continue')  # , justify="left")

    ok_button = tk.Button(info_window, text="OK", command=on_info_window_close)
    ok_button.grid(row=4, column=0, padx=10, pady=10)

    label_edge_length_info.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    label_tol_info.grid(row=1, column=0, padx=10, pady=10, sticky='w')
    label_info1.grid(row=2, column=0, padx=10, pady=10, sticky='w')
    label_info2.grid(row=3, column=0, padx=10, pady=10, sticky='w')
    label_info3.grid(row=4, column=0, padx=10, pady=10, sticky='w')
    label_info4.grid(row=5, column=0, padx=10, pady=10)
    ok_button.grid(row=6, column=0, padx=10, pady=10)

    # Changing font for info window interface elements
    label_edge_length_info.config(font=("Arial", BASE_FONT_SIZE))
    label_tol_info.config(font=("Arial", BASE_FONT_SIZE))
    label_info1.config(font=("Arial", BASE_FONT_SIZE))
    label_info2.config(font=("Arial", BASE_FONT_SIZE))
    label_info3.config(font=("Arial", BASE_FONT_SIZE))
    label_info4.config(font=("Arial", BASE_FONT_SIZE))
    label_custom_function3d.config(font=("Arial", BASE_FONT_SIZE))
    entry_custom_function3d.config(font=("Arial", BASE_FONT_SIZE))
    label_custom_function2d.config(font=("Arial", BASE_FONT_SIZE))
    entry_custom_function2d.config(font=("Arial", BASE_FONT_SIZE))
    label_custom_function4d.config(font=("Arial", BASE_FONT_SIZE))
    entry_custom_function4d.config(font=("Arial", BASE_FONT_SIZE))
    ok_button.config(font=("Arial", BASE_FONT_SIZE))


# ----------------------------------------------------------------------------------------------------------------------
def handle_2d_function(custom_function_str, initial_point, edge_length, tol):
    """
    Handles a 2D custom function by validating, plotting, and displaying errors if necessary.

    :param custom_function_str: string, a 2D custom function represented as a string
    :param initial_point: list, the initial point for the optimization algorithm
    :param edge_length: float, the edge length for the optimization algorithm
    :param tol: float, the tolerance for the optimization algorithm
    :return: None
    """

    if is_valid_function(custom_function_str, 2):
        x = sp.symbols("x")
        custom_function_str = replace_symbols_with_constants(custom_function_str)
        custom_function_expr = sp.sympify(custom_function_str)
        selected_function = sp.lambdify([(x,)], custom_function_expr, "numpy")

        function_variables = custom_function_expr.free_symbols

        if len(function_variables) == 1:
            plot_2d(selected_function, initial_point, edge_length, tol)
        else:
            showerror("Error", "2D custom function must have only 1 variable!")
    else:
        showerror("Error", "Please, enter a valid 2D custom function!")


# ----------------------------------------------------------------------------------------------------------------------
def handle_3d_function(custom_function_str, initial_point, edge_length, tol):
    """
    Handles a 3D custom function by validating, plotting, and displaying errors if necessary.

    :param custom_function_str: string, a 3D custom function represented as a string
    :param initial_point: list, the initial point for the optimization algorithm
    :param edge_length: float, the edge length for the optimization algorithm
    :param tol: float, the tolerance for the optimization algorithm
    :return: None
    """
    if is_valid_function(custom_function_str, 3):
        x, y = sp.symbols("x y")
        custom_function_str = replace_symbols_with_constants(custom_function_str)
        custom_function_expr = sp.sympify(custom_function_str)
        selected_function = sp.lambdify([(x, y)], custom_function_expr, "numpy")

        function_variables = custom_function_expr.free_symbols

        if len(function_variables) == 2:
            plot_3d(selected_function, initial_point, edge_length, tol)
        else:
            showerror("Error", "3D custom function must have 2 variables!")
    else:
        showerror("Error", "Please, enter a valid 3D custom function!")


# ----------------------------------------------------------------------------------------------------------------------
def handle_4d_function(custom_function_str, initial_point, edge_length, tol):
    """
    Handles a 4D custom function by validating, and displaying errors if necessary.

    :param custom_function_str: string, a 4D custom function represented as a string
    :param initial_point: list, the initial point for the optimization algorithm
    :param edge_length: float, the edge length for the optimization algorithm
    :param tol: float, the tolerance for the optimization algorithm
    :return: None
    """
    if is_valid_function(custom_function_str, 4):
        x, y, z = sp.symbols("x y z")
        custom_function_str = replace_symbols_with_constants(custom_function_str)
        custom_function_expr = sp.sympify(custom_function_str)
        selected_function = sp.lambdify([(x, y, z)], custom_function_expr, "numpy")

        function_variables = custom_function_expr.free_symbols

        if len(function_variables) == 3:
            show_4d(selected_function, initial_point, edge_length, tol)
        else:
            showerror("Error", "4D custom function must have 3 variables!")
    else:
        showerror("Error", "Please, enter a valid 4D custom function!")


# ----------------------------------------------------------------------------------------------------------------------
def submit():
    """
    Data processing after pressing the "submit" button. If all the data is correct,
    the algorithm is launched, the function and the extremum are drawn

    :return: None
    """
    try:
        initial_point = [float(entry_init_point_x.get()), float(entry_init_point_y.get()),
                         float(entry_init_point_z.get())]
        edge_length = float(entry_edge_length.get())
        tol = float(entry_tol.get())

        custom_function_str_2d = entry_custom_function2d.get()
        custom_function_str_3d = entry_custom_function3d.get()
        custom_function_str_4d = entry_custom_function4d.get()

        if not 1 <= edge_length <= 10:
            showerror("Error", "Edge length must be in the range [1, 10].")
            return

        if not 0.000001 <= tol <= 0.1:
            showerror("Error", "Tolerance must be in the range [0.000001, 0.1].")
            return

        custom_function_count = sum([bool(custom_function_str_2d),
                                     bool(custom_function_str_3d),
                                     bool(custom_function_str_4d)])

        if custom_function_count > 1:
            showerror("Error", "Please, enter only one custom function (either 2D, 3D, or 4D)!")
            return

        if custom_function_str_2d:
            handle_2d_function(custom_function_str_2d, initial_point, edge_length, tol)
        elif custom_function_str_3d:
            handle_3d_function(custom_function_str_3d, initial_point, edge_length, tol)
        elif custom_function_str_4d:
            handle_4d_function(custom_function_str_4d, initial_point, edge_length, tol)
        else:
            function_index = int(functions_name.index(combo_function.get()))
            selected_function = get_function_by_index(function_index)
            if function_index in [0, 1, 2]:
                plot_2d(selected_function, initial_point, edge_length, tol)
            elif function_index in [3, 4, 5]:
                plot_3d(selected_function, initial_point, edge_length, tol)
            else:
                show_4d(selected_function, initial_point, edge_length, tol)

    except ValueError:
        showerror("Error", "Please, enter valid parameters!")
        return


# ----------------------------------------------------------------------------------------------------------------------
BASE_FONT_SIZE = 16
root = tk.Tk()
root.minsize(520, 320)  # Minimum size of the window
root.maxsize(800, 600)  # Maximum size of the window
root.title("Nelder-Mead Parameters")
root.withdraw()  # hide the main window
# root.geometry("520x320")
# Configure the grid
for i in range(10):
    root.grid_columnconfigure(i, weight=1)
    root.grid_rowconfigure(i, weight=1)

# Get screen size
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set window size as a percentage of screen size
window_width = int(screen_width * 0.3)  # 30% of screen width
window_height = int(screen_height * 0.4)  # 40% of screen height

# Set the geometry of the window
root.geometry(f"{window_width}x{window_height}")

# Creation of interface elements
default_init_point_x = "2"
default_init_point_y = "1"
default_init_point_z = "1"
default_edge_length = "1"
default_tol = "0.000001"

label_function = tk.Label(root, text="Function:")
style = ttk.Style()
style.configure("TCombobox", font=("Arial", BASE_FONT_SIZE))
combo_function = ttk.Combobox(root, values=functions_name, state="readonly", width=30)

label_init_point_x = tk.Label(root, text="Initial Point X:")
entry_init_point_x = tk.Entry(root)
label_init_point_y = tk.Label(root, text="Initial Point Y:")
entry_init_point_y = tk.Entry(root)
label_init_point_z = tk.Label(root, text="Initial Point Z:")
entry_init_point_z = tk.Entry(root)
label_edge_length = tk.Label(root, text="Edge Length:")
entry_edge_length = tk.Entry(root)
label_tol = tk.Label(root, text="Tolerance:")
entry_tol = tk.Entry(root)
label_custom_function3d = tk.Label(root, text="Custom 3d Function:")
entry_custom_function3d = tk.Entry(root)
label_custom_function2d = tk.Label(root, text="Custom 2d Function:")
entry_custom_function2d = tk.Entry(root)
label_custom_function4d = tk.Label(root, text="Custom 4d Function:")
entry_custom_function4d = tk.Entry(root)
submit_button = tk.Button(root, text="Submit", command=submit)

entry_init_point_x.insert(0, default_init_point_x)
entry_init_point_y.insert(0, default_init_point_y)
entry_init_point_z.insert(0, default_init_point_y)
entry_edge_length.insert(0, default_edge_length)
entry_tol.insert(0, default_tol)

# Changing the font for all interface elements
label_function.config(font=("Arial", BASE_FONT_SIZE))
combo_function.config(font=("Arial", BASE_FONT_SIZE))
label_init_point_x.config(font=("Arial", BASE_FONT_SIZE))
entry_init_point_x.config(font=("Arial", BASE_FONT_SIZE))
label_init_point_y.config(font=("Arial", BASE_FONT_SIZE))
entry_init_point_y.config(font=("Arial", BASE_FONT_SIZE))
label_init_point_z.config(font=("Arial", BASE_FONT_SIZE))
entry_init_point_z.config(font=("Arial", BASE_FONT_SIZE))
label_edge_length.config(font=("Arial", BASE_FONT_SIZE))
entry_edge_length.config(font=("Arial", BASE_FONT_SIZE))
label_tol.config(font=("Arial", BASE_FONT_SIZE))
entry_tol.config(font=("Arial", BASE_FONT_SIZE))
submit_button.config(font=("Arial", BASE_FONT_SIZE))

show_info_window()
# Run the main event loop
root.mainloop()
# ----------------------------------------------------------------------------------------------------------------------
