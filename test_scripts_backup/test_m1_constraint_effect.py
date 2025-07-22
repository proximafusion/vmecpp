#!/usr/bin/env python3
"""Test the effect of M=1 constraint on symmetric case."""



# Original formula with scaling_factor = 0.5
def original_m1_constraint(rss, zcs):
    """Original rotation formula."""
    backup_rss = rss
    new_rss = (backup_rss + zcs) * 0.5
    new_zcs = (backup_rss - zcs) * 0.5
    return new_rss, new_zcs


# New formula (averaging)
def new_m1_constraint(rss, zcs):
    """New averaging formula."""
    constrained_value = (rss + zcs) / 2.0
    return constrained_value, constrained_value


# Test with circular tokamak values
rss_input = 2.0  # rbss[1,0]
zcs_input = 2.0  # zbcs[1,0]

print("Circular tokamak symmetric case:")
print(f"Input: rbss[1,0] = {rss_input}, zbcs[1,0] = {zcs_input}")

# Apply original formula
rss_old, zcs_old = original_m1_constraint(rss_input, zcs_input)
print("\nOriginal formula (rotation with factor 0.5):")
print(f"  rbss[1,0] = ({rss_input} + {zcs_input}) * 0.5 = {rss_old}")
print(f"  zbcs[1,0] = ({rss_input} - {zcs_input}) * 0.5 = {zcs_old}")

# Apply new formula
rss_new, zcs_new = new_m1_constraint(rss_input, zcs_input)
print("\nNew formula (averaging):")
print(f"  rbss[1,0] = ({rss_input} + {zcs_input}) / 2.0 = {rss_new}")
print(f"  zbcs[1,0] = ({rss_input} + {zcs_input}) / 2.0 = {zcs_new}")

print("\nDifference:")
print(f"  rbss[1,0]: {rss_old} -> {rss_new} (change: {rss_new - rss_old})")
print(f"  zbcs[1,0]: {zcs_old} -> {zcs_new} (change: {zcs_new - zcs_old})")

# Test with unequal values
print("\n" + "=" * 50)
print("Test with unequal values (to see the difference):")
rss_test = 3.0
zcs_test = 1.0
print(f"Input: rbss[1,0] = {rss_test}, zbcs[1,0] = {zcs_test}")

rss_old2, zcs_old2 = original_m1_constraint(rss_test, zcs_test)
print("\nOriginal formula:")
print(f"  rbss[1,0] = {rss_old2}")
print(f"  zbcs[1,0] = {zcs_old2}")

rss_new2, zcs_new2 = new_m1_constraint(rss_test, zcs_test)
print("\nNew formula:")
print(f"  rbss[1,0] = {rss_new2}")
print(f"  zbcs[1,0] = {zcs_new2}")
