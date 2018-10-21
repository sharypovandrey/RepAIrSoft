from app import models
from django.contrib import admin

class Actions(admin.ModelAdmin):
    model = models.Actions
admin.site.register(models.Actions, Actions)

class Equipment(admin.ModelAdmin):
    model = models.Equipment
admin.site.register(models.Project, Equipment)

class TechnicalPerson(admin.ModelAdmin):
    model = models.TechnicalPerson
admin.site.register(models.TechnicalPerson, TechnicalPerson)

class Building(admin.ModelAdmin):
    model = models.Building
admin.site.register(models.Building, Building)

class Issues(admin.ModelAdmin):
    model = models.Issues
admin.site.register(models.Issues, Issues)

class Operation(admin.ModelAdmin):
    model = models.Operation
admin.site.register(models.Operation, Operation)