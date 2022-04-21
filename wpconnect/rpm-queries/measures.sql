select
      m.id as measure_id
    , md.member_id
    , md.measure
    , md.value
    , md.value_numeric
    , m.type as base_measure_type
    , m.measured_at
    , m.measured_at_tz_offset
    , m.measure_day_et
    , m.measure_epoch
    , mem.first_name
    , mem.last_name
    , mem.extern_id
    , mem.email
    , mem.first_name || ' ' || mem.last_name as full_name
    , left(mem.first_name, 1) || left(mem.last_name, 1) as initials
    , m.updated_at
    , m.updated_at_tz_offset
from rpm.measure_details md
inner join rpm.measures m on m.id = md.measure_id
left join rpm.members mem on mem.id = md.member_id
where mem.last_name not like '%%test%%'
order by member_id, measured_at