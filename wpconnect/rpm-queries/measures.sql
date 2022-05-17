with meas AS (
  select
      m.id as measure_id
      , md.member_id
      , md.measure
      , md.value
      , md.value_numeric
      , lag(md.value_numeric) over(partition by md.member_id, md.measure order by m.measured_at) as last_value_numeric
      , m.type as base_measure_type
      , m.measured_at
      , m.measured_at_tz_offset
      , m.measure_day_et
      , m.measure_epoch
      , case when
          mem.program = '' then 'None'
          else coalesce(mem.program, 'None')
          end as program
      , DATE_PART('day', m.measured_at - to_timestamp(bt.enroll_epoch)) as days_since_enrollment
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
  left join (
		select distinct member_id, enroll_epoch from rpm.billing_tally
	) bt on bt.member_id = mem.id
  where mem.last_name not like '%%test%%'
  order by md.member_id, md.measure, m.measured_at
)

select
    meas.*
  , meas.value_numeric / nullif(meas.last_value_numeric, 0) - 1 as delta_from_last
from meas
where 1=1
  and value_numeric is not null
