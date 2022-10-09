from encodings import utf_8
import os
import dateutil.parser
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class Model():

    def __init__(self, ingredients: list):
        self.ingredients = set(ingredients)
        pd.options.mode.chained_assignment = None
        self.total_recipes = pd.DataFrame([eval(line) for line in open(r'testing/src/Backend/trainRecipes.json', encoding="utf8")])
        self.total_recipes
        self.ingredients_per_recipes = self.total_recipes.explode("ingredients").groupby("recipe_id")["ingredients"].apply(set).to_dict()
        self.recipes_per_ingredients = self.total_recipes.explode("ingredients").groupby("ingredients")["recipe_id"].apply(set).to_dict()
        self.instructions = self.total_recipes.explode("ingredients").groupby("ingredients")["steps"].apply(set).to_dict()

    

    def Jaccard(self, s1, s2):
        numer = len(s1.intersection(s2))
        denom = len(s1.union(s2))
        if denom == 0:
            return 0
        return numer / denom


    def most_similar_recipe(self, data, ingredients, ingredients_per_recipes, n=5):
        recipes = list(ingredients_per_recipes.keys())
        steps = list(self.instructions.keys())
        tuples = []
        for recipe in recipes:
            curr_ingredients = ingredients_per_recipes[recipe]
            max_list = []
            for ingredient in ingredients:
                ingredient_sims = []
                for curr in curr_ingredients:
                    ingredient_sims.append(self.Jaccard(set(ingredient), set(curr)))
                max_sim = max(ingredient_sims)
                max_list.append(max_sim)
            sim = np.mean(max_list)
            tuples.append((sim, recipe))

        combined = list(sorted(tuples, reverse=True))
        return combined[:n]

    def generated_recipe(self):
        ingredients_per_recipes = self.total_recipes.explode("ingredients").groupby("name")["ingredients"].apply(set).to_dict()
        #ingredients_per_instructions = self.total_recipes.explode("ingredients").groupby("steps")["ingredients"].apply(set).to_dict()
        return self.most_similar_recipe(self.total_recipes.copy(), self.ingredients, ingredients_per_recipes)

    example1 = set(["dark chocolate cake mix", "dark chocolate chips", "flour","eggs"])
    example2 = set(["salt", "pepper", "salmon"])
    example3 = set(["cinnamon", "cherries", "butterscotch", "vodka"])


recommendation = Model(['pasta', 'salsa', 'tortillas', 'beans']).generated_recipe()
print(recommendation)
