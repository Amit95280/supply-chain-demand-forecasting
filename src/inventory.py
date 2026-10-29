
import math

def safety_stock(demand_std, lead_time_days, service_factor=1.65):
    return max(0.0, service_factor * demand_std * math.sqrt(max(1, lead_time_days)))

def reorder_point(avg_daily_demand, lead_time_days, safety_stock_value):
    return max(0.0, avg_daily_demand * lead_time_days + safety_stock_value)

def inventory_status(current_inventory, reorder_point_value):
    if current_inventory <= reorder_point_value:
        return "REORDER NOW"
    if current_inventory <= reorder_point_value * 1.25:
        return "WATCH"
    return "HEALTHY"
