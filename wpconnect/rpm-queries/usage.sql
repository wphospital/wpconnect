select
      u.member_id
    , mem.first_name
    , mem.last_name
    , mem.extern_id
    , mem.email
    , mem.first_name || ' ' || mem.last_name as full_name
    , u.last_measured_at
    , current_date - u.last_measured_at::DATE as days_since_last_measurement
from rpm.usage u
left join rpm.members mem on mem.id = u.member_id
where mem.last_name not like '%%test%%'
order by u.last_measured_at desc
