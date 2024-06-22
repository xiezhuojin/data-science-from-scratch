from typing import List

import matplotlib.pyplot as plt
import tqdm

from scratch.linear_algebra.vectors import Vector, subtract, vector_mean, \
    magnitude, dot, scaler_multiply
from scratch.gradient_descent.using_the_gradient import gradient_step


pca_data = [
    [20.9666776351559,-13.1138080189357],
    [22.7719907680008,-19.8890894944696],
    [25.6687103160153,-11.9956004517219],
    [18.0019794950564,-18.1989191165133],
    [21.3967402102156,-10.8893126308196],
    [0.443696899177716,-19.7221132386308],
    [29.9198322142127,-14.0958668502427],
    [19.0805843080126,-13.7888747608312],
    [16.4685063521314,-11.2612927034291],
    [21.4597664701884,-12.4740034586705],
    [3.87655283720532,-17.575162461771],
    [34.5713920556787,-10.705185165378],
    [13.3732115747722,-16.7270274494424],
    [20.7281704141919,-8.81165591556553],
    [24.839851437942,-12.1240962157419],
    [20.3019544741252,-12.8725060780898],
    [21.9021426929599,-17.3225432396452],
    [23.2285885715486,-12.2676568419045],
    [28.5749111681851,-13.2616470619453],
    [29.2957424128701,-14.6299928678996],
    [15.2495527798625,-18.4649714274207],
    [26.5567257400476,-9.19794350561966],
    [30.1934232346361,-12.6272709845971],
    [36.8267446011057,-7.25409849336718],
    [32.157416823084,-10.4729534347553],
    [5.85964365291694,-22.6573731626132],
    [25.7426190674693,-14.8055803854566],
    [16.237602636139,-16.5920595763719],
    [14.7408608850568,-20.0537715298403],
    [6.85907008242544,-18.3965586884781],
    [26.5918329233128,-8.92664811750842],
    [-11.2216019958228,-27.0519081982856],
    [8.93593745011035,-20.8261235122575],
    [24.4481258671796,-18.0324012215159],
    [2.82048515404903,-22.4208457598703],
    [30.8803004755948,-11.455358009593],
    [15.4586738236098,-11.1242825084309],
    [28.5332537090494,-14.7898744423126],
    [40.4830293441052,-2.41946428697183],
    [15.7563759125684,-13.5771266003795],
    [19.3635588851727,-20.6224770470434],
    [13.4212840786467,-19.0238227375766],
    [7.77570680426702,-16.6385739839089],
    [21.4865983854408,-15.290799330002],
    [12.6392705930724,-23.6433305964301],
    [12.4746151388128,-17.9720169566614],
    [23.4572410437998,-14.602080545086],
    [13.6878189833565,-18.9687408182414],
    [15.4077465943441,-14.5352487124086],
    [20.3356581548895,-10.0883159703702],
    [20.7093833689359,-12.6939091236766],
    [11.1032293684441,-14.1383848928755],
    [17.5048321498308,-9.2338593361801],
    [16.3303688220188,-15.1054735529158],
    [26.6929062710726,-13.306030567991],
    [34.4985678099711,-9.86199941278607],
    [39.1374291499406,-10.5621430853401],
    [21.9088956482146,-9.95198845621849],
    [22.2367457578087,-17.2200123442707],
    [10.0032784145577,-19.3557700653426],
    [14.045833906665,-15.871937521131],
    [15.5640911917607,-18.3396956121887],
    [24.4771926581586,-14.8715313479137],
    [26.533415556629,-14.693883922494],
    [12.8722580202544,-21.2750596021509],
    [24.4768291376862,-15.9592080959207],
    [18.2230748567433,-14.6541444069985],
    [4.1902148367447,-20.6144032528762],
    [12.4332594022086,-16.6079789231489],
    [20.5483758651873,-18.8512560786321],
    [17.8180560451358,-12.5451990696752],
    [11.0071081078049,-20.3938092335862],
    [8.30560561422449,-22.9503944138682],
    [33.9857852657284,-4.8371294974382],
    [17.4376502239652,-14.5095976075022],
    [29.0379635148943,-14.8461553663227],
    [29.1344666599319,-7.70862921632672],
    [32.9730697624544,-15.5839178785654],
    [13.4211493998212,-20.150199857584],
    [11.380538260355,-12.8619410359766],
    [28.672631499186,-8.51866271785711],
    [16.4296061111902,-23.3326051279759],
    [25.7168371582585,-13.8899296143829],
    [13.3185154732595,-17.8959160024249],
    [3.60832478605376,-25.4023343597712],
    [39.5445949652652,-11.466377647931],
    [25.1693484426101,-12.2752652925707],
    [25.2884257196471,-7.06710309184533],
    [6.77665715793125,-22.3947299635571],
    [20.1844223778907,-16.0427471125407],
    [25.5506805272535,-9.33856532270204],
    [25.1495682602477,-7.17350567090738],
    [15.6978431006492,-17.5979197162642],
    [37.42780451491,-10.843637288504],
    [22.974620174842,-10.6171162611686],
    [34.6327117468934,-9.26182440487384],
    [34.7042513789061,-6.9630753351114],
    [15.6563953929008,-17.2196961218915],
    [25.2049825789225,-14.1592086208169]
]

x = [d[0] for d in pca_data]
y = [d[1] for d in pca_data]
plt.scatter(x, y)
plt.show()

# Sometimes the "actual" (or useful) dimensions of the data might not correspond 
# to the dimensions we have. When this is the case, we can use a techinique called 
# principao component analysis (PCA) to extract one or more dimensions that capture 
# as much of the variation in the data as possible.

# As a first step, we'll need to translate the data so that each dimension has 
# mean 0:

def de_mean(data: List[Vector]) -> List[Vector]:
    """
    Recenters the data to have mean 0 in every dimension.
    """

    mean = vector_mean(data)
    return [subtract(d, mean) for d in data]

de_meanded_pca_data = de_mean(pca_data)
x = [d[0] for d in de_meanded_pca_data]
y = [d[1] for d in de_meanded_pca_data]
plt.scatter(x, y)
plt.show()

# Now, given a de-meanded matrix X, we can ask which is the direction that captures 
# the greatest variance in the data.

# Specifically, given a direction d (a vector of magnitude 1), each row x in the 
# matrix extends dot(x, d) in the d direction. And every nonzero vector w determines 
# a direction if we rescale it to have magniude 1;

def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]

# Therefore, given a nonzero vector w, we can compute the variance of our dataset 
# in the direction determined by w:

def directional_variance(data: List[Vector], w: Vector) -> float:
    """
    Returns the variance of x in the direction of w.
    """

    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)

# We'd like to find the direction that maximizes this variance. We can do this 
# using gradient descent, as soon as we have the gradient function:

def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """
    The gradient of directional variance with respect to w.
    """

    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data)
            for i in range(len(w))]

# And now the first principal component that we have is just the direction that 
# maximizes the directional_variance function:

def first_principal_component(data: List[Vector],
                              n: int=100,
                              step_size: float=0.1) -> Vector:
    # Start with a random guess
    guess = [1.0 for _ in data[0]]

    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")

    return direction(guess)

de_meanded_pca_data = de_mean(de_meanded_pca_data)
first_pc = first_principal_component(de_meanded_pca_data)
x = [d[0] for d in de_meanded_pca_data]
y = [d[1] for d in de_meanded_pca_data]
plt.scatter(x, y)
plt.arrow(0, 0, first_pc[0], first_pc[1])
plt.show()

# Once we've found the direction that this's the first principal component, we 
# can project our data onto it to find the values of that component:

def project(v: Vector, w: Vector) -> Vector:
    """
    Return the projection of v onto the direction w.
    """

    project_length = dot(v, w)
    return scaler_multiply(project_length, w)

# If we want to find further component, we first remove the projections from the 
# data

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """
    projects v onto w and subtracts the result from v.
    """

    return subtract(v, project(v, w))

def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]

data_after_removed_1pc = remove_projection(de_meanded_pca_data, first_pc)
plt.scatter([d[0] for d in data_after_removed_1pc], [d[1] for d in data_after_removed_1pc])
plt.show()

# At that point, we can find the next principal component by repreating the process 
# on the result of remove_projection
second_pc = first_principal_component(data_after_removed_1pc)

# On a higher-dimensional dataset, we can iteratively find as many components as 
# we want

def pca(data: List[Vector], num_component: int) -> List[Vector]:
    components: List[Vector] = []
    for _ in num_component:
        component = first_principal_component(data)
        components.append(component)
        data = [remove_projection_from_vector(v, component) for v in data]

    return components

# same as above (but don't forget to standardlize your data)
# from statsmodels.multivariate.pca import PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_ = pca.fit(pca_data)
print(pca_.explained_variance_ratio_)

# We can then transform our data into the lower-dimensional space spanned by the 
# components:

def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, component) for component in components]

def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data]

# This technique is valueable for a couple of reasons. First, it can help us 
# clean our data by eliminating noise dimensions and consolidating highly correlated 
# dimensions. Second, after extracting a low-dimensional representation of our 
# data, we can use a variety of techniques that don't work as well on high-dimensional 
# data.