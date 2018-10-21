"""
Definition of models.
"""

from django.db import models

# Create your models here.

class TechnicalPerson(models.Model):
    name = models.CharField(max_length=50, blank=True, null=True)
    def __str__(self):
        return self.name

class Building(models.Model):
    name = models.CharField(max_length=50, blank=True, null=True)
    def __str__(self):
        return self.name

class Actions(models.Model):
    name = models.CharField(max_length=50, blank=True, null=True)
    def __str__(self):
        return self.name

class Issues(models.Model):
    name = models.CharField(max_length=50, blank=True, null=True)
    def __str__(self):
        return self.name

EQUIPMENT_TYPE = (
        ("ac", "AIR COOLING"),
        ("wm", "WASHING MACHINE"),
        ("light", "LIGHTING"),
        ("wts", "WATER TREATMENT SYSTEM")
    )

class Equipment(models.Model):
    type = models.CharField(choices=EEQUIPMENT_TYPE, blank=True, null=True)
    model = models.CharField(choices=MODEL, blank=True, null=True)
    bought = models.DateField(blank=True, null=True)
    qr_code = models.CharField(max_lenght=550 ,blank=True, null=True)
    building = models.ForeignField(Building, blank=True, null=True)

    def __str__(self):
        return self.name

class Operation(models.Model):
    equipment = models.ForeignField(Equipment, blank=True, null=True)
    action = models.ForeignField(Equipment, blank=True, null=True)
    person = models.ForeignField(Equipment, blank=True, null=True)
    date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.equipment.model + " " + self.action.name