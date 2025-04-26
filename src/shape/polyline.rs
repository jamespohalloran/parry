use crate::bounding_volume::Aabb;
use crate::math::{Isometry, Point, Real, Vector};
use crate::partitioning::Qbvh;
use crate::query::{PointProjection, PointQueryWithLocation};
use crate::shape::composite_shape::SimdCompositeShape;
use crate::shape::{FeatureId, Segment, SegmentPointLocation, Shape, TypedSimdCompositeShape};
use crate::query::details::NormalConstraints;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize),
    archive(check_bytes)
)]
pub struct Polyline {
    qbvh: Qbvh<u32>,
    vertices: Vec<Point<Real>>,
    indices: Vec<[u32; 2]>,
}

#[derive(Clone, Debug)]
pub struct SimpleNormalSnapper;

impl NormalConstraints for SimpleNormalSnapper {
    fn project_local_normal_mut(&self, normal: &mut Vector<Real>) -> bool {
        let abs_x = normal.x.abs();
        let abs_y = normal.y.abs();

        if abs_x > abs_y {
            normal.x = normal.x.signum();
            normal.y = 0.0;
        } else {
            normal.x = 0.0;
            normal.y = normal.y.signum();
        }

        true
    }
}

impl Polyline {
    pub fn new(vertices: Vec<Point<Real>>, indices: Option<Vec<[u32; 2]>>) -> Self {
        let indices = indices.unwrap_or_else(|| (0..vertices.len() as u32 - 1).map(|i| [i, i + 1]).collect());
        let data = indices.iter().enumerate().map(|(i, idx)| {
            let aabb = Segment::new(vertices[idx[0] as usize], vertices[idx[1] as usize]).local_aabb();
            (i as u32, aabb)
        });

        let mut qbvh = Qbvh::new();
        qbvh.clear_and_rebuild(data, 0.0);

        Self { qbvh, vertices, indices }
    }

    pub fn aabb(&self, pos: &Isometry<Real>) -> Aabb {
        self.qbvh.root_aabb().transform_by(pos)
    }

    pub(crate) fn qbvh(&self) -> &Qbvh<u32> {
        &self.qbvh
    }

    pub fn num_segments(&self) -> usize {
        self.indices.len()
    }

    pub fn segments(&self) -> impl ExactSizeIterator<Item = Segment> + '_ {
        self.indices.iter().map(move |ids| {
            Segment::new(self.vertices[ids[0] as usize], self.vertices[ids[1] as usize])
        })
    }

    pub fn segment(&self, i: u32) -> Segment {
        let idx = self.indices[i as usize];
        Segment::new(self.vertices[idx[0] as usize], self.vertices[idx[1] as usize])
    }

    pub fn vertices(&self) -> &[Point<Real>] {
        &self.vertices
    }

    pub fn indices(&self) -> &[[u32; 2]] {
        &self.indices
    }
}

impl SimdCompositeShape for Polyline {
    fn map_part_at(
        &self,
        i: u32,
        f: &mut dyn FnMut(Option<&Isometry<Real>>, &dyn Shape, Option<&dyn NormalConstraints>),
    ) {
        let segment = self.segment(i);
        f(None, &segment, Some(&SimpleNormalSnapper));
    }

    fn qbvh(&self) -> &Qbvh<u32> {
        &self.qbvh
    }
}

impl TypedSimdCompositeShape for Polyline {
    type PartShape = Segment;
    type PartNormalConstraints = SimpleNormalSnapper;
    type PartId = u32;

    fn map_typed_part_at(
        &self,
        i: u32,
        mut f: impl FnMut(
            Option<&Isometry<Real>>,
            &Self::PartShape,
            Option<&Self::PartNormalConstraints>,
        ),
    ) {
        let segment = self.segment(i);
        f(None, &segment, Some(&SimpleNormalSnapper));
    }

    fn map_untyped_part_at(
        &self,
        i: u32,
        mut f: impl FnMut(Option<&Isometry<Real>>, &dyn Shape, Option<&dyn NormalConstraints>),
    ) {
        let segment = self.segment(i);
        f(None, &segment, Some(&SimpleNormalSnapper));
    }

    fn typed_qbvh(&self) -> &Qbvh<u32> {
        &self.qbvh
    }
}
